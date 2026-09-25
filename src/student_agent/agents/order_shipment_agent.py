"""Order, product and shipment specialist.

Calls only tools discovered for this case. Returns a shipment verdict and candidate
causes. The coordinator owns the final root cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..models.messages import CandidateCause, EntityResult, ShipmentResult, ShipmentVerdict
from ..trace import TraceWriter
from ..utils.cache import CaseCache
from ..utils.evidence import (
    CARRIER_KEYS,
    CUSTOMER_DELIVERY_KEYS,
    ESTIMATED_DELIVERY_KEYS,
    SHIPPING_LIMIT_KEYS,
    as_float,
    as_naive,
    evidence_ref_of,
    first_datetime,
    first_text,
    unique_ids,
    walk_dicts,
)

ACTOR = "order-shipment-agent"
COORDINATOR = "coordinator"
MAX_ATTEMPTS = 2

ITEM_TOOL = "get_order_items"
SHIPMENT_TOOL = "get_shipment_summary"
ORDER_TOOL = "get_order"
PRODUCT_TOOL = "get_product_context"

LOST = {"lost", "missing", "shipment_lost"}
RETURNED = {"returned", "returning", "returned_to_seller", "return_to_sender"}
DELIVERED = {"delivered"}


@dataclass
class OrderShipmentResult(ShipmentResult):
    """ShipmentResult plus facts the coordinator needs for conflict checks."""

    order_total_brl: float | None = None
    order_status: str | None = None
    shipment_status: str | None = None
    conflicts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _OrderView:
    verdict: ShipmentVerdict
    late_sellers: list[str]
    timeline_complete: bool
    item_ids: list[str]
    seller_ids: list[str]
    shipment_ids: list[str]
    order_total_brl: float | None
    order_status: str | None
    shipment_status: str | None
    conflicts: list[dict[str, Any]]


class _Session:
    """One case: fixed case_id, cache, and a two-attempt transport retry."""

    def __init__(self, case_id: str, gateway: EvidenceGateway, tools: set[str]) -> None:
        self.case_id = case_id
        self.gateway = gateway
        self.tools = tools
        self.cache = CaseCache()

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.tools

    async def call(self, tool_name: str, **arguments: str) -> dict[str, Any] | None:
        hit, cached = self.cache.get(self.case_id, tool_name, arguments)
        if hit:
            return cached
        evidence: dict[str, Any] | None = None
        if self.allows(tool_name):
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    evidence = await self.gateway.call(
                        tool_name, case_id=self.case_id, **arguments
                    )
                    break
                except (RuntimeError, ValueError):
                    break
                except Exception:
                    if attempt == MAX_ATTEMPTS:
                        break
        self.cache.put(self.case_id, tool_name, arguments, evidence)
        return evidence


def _normalize(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _kind(value: str | None) -> str | None:
    text = _normalize(value)
    if text is None:
        return None
    if text in LOST or text.endswith("_lost"):
        return "lost"
    if "return" in text:
        return "returned"
    if text in DELIVERED or text.endswith("_delivered"):
        return "delivered"
    return text


def _item_rows(data: Any) -> list[dict[str, Any]]:
    keys = ("order_item_id", "item_id", "seller_id", "product_id", "price", "shipping_limit_date")
    rows: list[dict[str, Any]] = []
    for node in walk_dicts(data):
        if any(key in node and not isinstance(node.get(key), dict | list) for key in keys):
            rows.append(node)
    return rows


def _status_flags(data: Any) -> set[str]:
    flags: set[str] = set()
    for node in walk_dicts(data):
        for key in ("shipment_status", "event_type", "type", "status"):
            kind = _kind(str(node[key])) if key in node and node[key] is not None else None
            if kind in {"lost", "returned", "delivered"}:
                flags.add(kind)
    return flags


def _after(left: datetime | None, right: datetime | None) -> bool:
    if left is None or right is None:
        return False
    return as_naive(left) > as_naive(right)


def _analyze(
    order_data: Any, item_data: Any, shipment_data: Any, *, opened_at: datetime | None
) -> _OrderView:
    items = _item_rows(item_data)
    item_ids = unique_ids(
        [row.get("item_id") or row.get("order_item_id") or row.get("product_id") for row in items]
    )
    seller_ids = unique_ids([row.get("seller_id") for row in items])
    shipment_ids = unique_ids(
        [node.get("shipment_id") for node in walk_dicts(shipment_data)]
    )
    order_status = first_text(order_data, ("order_status",))
    shipment_status = first_text(shipment_data, ("shipment_status", "status"))
    flags = _status_flags(shipment_data)
    ship_kind = _kind(shipment_status)
    order_kind = _kind(order_status)

    carrier = first_datetime(shipment_data, CARRIER_KEYS) or first_datetime(
        order_data, CARRIER_KEYS
    )
    customer_at = first_datetime(shipment_data, CUSTOMER_DELIVERY_KEYS) or first_datetime(
        order_data, CUSTOMER_DELIVERY_KEYS
    )
    estimated = first_datetime(shipment_data, ESTIMATED_DELIVERY_KEYS) or first_datetime(
        order_data, ESTIMATED_DELIVERY_KEYS
    )

    limits: list[tuple[str | None, datetime]] = []
    for row in items:
        limit = first_datetime(row, SHIPPING_LIMIT_KEYS)
        seller = row.get("seller_id")
        seller_id = str(seller).strip() if seller else None
        if limit is not None:
            limits.append((seller_id, limit))
    if not limits:
        shared_limit = first_datetime(shipment_data, SHIPPING_LIMIT_KEYS) or first_datetime(
            order_data, SHIPPING_LIMIT_KEYS
        )
        if shared_limit is not None:
            holders = seller_ids or [None]
            limits = [(seller, shared_limit) for seller in holders]

    late_sellers = unique_ids(
        [seller for seller, limit in limits if seller and _after(carrier, limit)]
    )
    unnamed_late = any(seller is None and _after(carrier, limit) for seller, limit in limits)
    seller_late = bool(late_sellers) or unnamed_late
    seller_unknown = bool(limits) and carrier is None and ship_kind not in {"lost", "returned"}
    logistics_late = (not seller_late) and _after(customer_at, estimated)
    if (
        not seller_late
        and customer_at is None
        and estimated is not None
        and opened_at is not None
        and ship_kind in {"shipped", "in_transit", "handling"}
        and _after(opened_at, estimated)
    ):
        logistics_late = True

    total = 0.0
    priced = False
    for row in items:
        price = as_float(row.get("price"))
        if price is None:
            continue
        priced = True
        total += price + (as_float(row.get("freight_value")) or 0.0)

    conflicts: list[dict[str, Any]] = []
    shipment_failed = ship_kind in {"lost", "returned"} or bool(flags & {"lost", "returned"})
    if order_kind == "delivered" and shipment_failed:
        conflicts.append(
            {
                "field": "shipment_status",
                "sources": ["order", "shipment"],
                "selected_source": "shipment",
                "resolution_code": "SHIPMENT_SOURCE_PRECEDENCE",
            }
        )

    clocks_present = customer_at is not None and estimated is not None
    timeline_complete = clocks_present and (not limits or carrier is not None)
    if "lost" in flags or ship_kind == "lost":
        verdict: ShipmentVerdict = "lost"
    elif "returned" in flags or ship_kind == "returned":
        verdict = "returned"
    elif seller_late and late_sellers:
        verdict = "seller_delay"
    elif seller_late or seller_unknown:
        verdict = "insufficient_evidence"
        timeline_complete = False
    elif logistics_late:
        verdict = "logistics_delay"
        if customer_at is None:
            timeline_complete = False
    elif (
        customer_at is not None
        and estimated is not None
        and not _after(customer_at, estimated)
        and (ship_kind == "delivered" or order_kind == "delivered" or "delivered" in flags)
    ):
        verdict = "on_time"
        timeline_complete = timeline_complete and not seller_unknown
    elif conflicts:
        verdict = "conflicting"
        timeline_complete = False
    else:
        verdict = "insufficient_evidence"
        timeline_complete = False

    if verdict == "insufficient_evidence":
        timeline_complete = False

    return _OrderView(
        verdict=verdict,
        late_sellers=late_sellers if verdict == "seller_delay" else [],
        timeline_complete=timeline_complete,
        item_ids=item_ids,
        seller_ids=seller_ids,
        shipment_ids=shipment_ids,
        order_total_brl=round(total, 2) if priced else None,
        order_status=order_status,
        shipment_status=shipment_status,
        conflicts=conflicts,
    )


def _aggregate(views: list[_OrderView]) -> _OrderView:
    def pick(kind: ShipmentVerdict) -> bool:
        return any(view.verdict == kind for view in views)

    if pick("lost"):
        verdict: ShipmentVerdict = "lost"
    elif pick("returned"):
        verdict = "returned"
    elif pick("seller_delay"):
        verdict = "seller_delay"
    elif pick("logistics_delay"):
        verdict = "logistics_delay"
    elif pick("conflicting"):
        verdict = "conflicting"
    elif views and all(view.verdict == "on_time" for view in views):
        verdict = "on_time"
    else:
        verdict = "insufficient_evidence"

    late = unique_ids([seller for view in views for seller in view.late_sellers])
    if verdict != "seller_delay":
        late = []
    statuses = {view.order_status for view in views if view.order_status}
    ship_statuses = {view.shipment_status for view in views if view.shipment_status}
    totals = [view.order_total_brl for view in views if view.order_total_brl is not None]
    conflicts: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for view in views:
        for conflict in view.conflicts:
            key = (conflict["field"], tuple(conflict["sources"]))
            if key not in seen:
                seen.add(key)
                conflicts.append(conflict)
    return _OrderView(
        verdict=verdict,
        late_sellers=late,
        timeline_complete=(
            bool(views)
            and all(view.timeline_complete for view in views)
            and verdict != "insufficient_evidence"
        ),
        item_ids=unique_ids([item for view in views for item in view.item_ids]),
        seller_ids=unique_ids([seller for view in views for seller in view.seller_ids]),
        shipment_ids=unique_ids([shipment for view in views for shipment in view.shipment_ids]),
        order_total_brl=round(sum(totals), 2) if totals else None,
        order_status=next(iter(statuses)) if len(statuses) == 1 else None,
        shipment_status=next(iter(ship_statuses)) if len(ship_statuses) == 1 else None,
        conflicts=conflicts,
    )


def _causes(view: _OrderView) -> list[CandidateCause]:
    seller = view.late_sellers[0] if view.late_sellers else None
    if view.verdict == "lost":
        return [CandidateCause("SHIPMENT_LOST", "logistics_provider", rank=1)]
    if view.verdict == "returned":
        return [CandidateCause("SHIPMENT_RETURNED", "logistics_provider", rank=1)]
    if view.verdict == "seller_delay":
        return [CandidateCause("SELLER_SHIPMENT_DELAY", "seller", seller, rank=1)]
    if view.verdict == "logistics_delay":
        return [CandidateCause("LOGISTICS_DELIVERY_DELAY", "logistics_provider", rank=1)]
    if view.verdict == "conflicting":
        return [CandidateCause("SHIPMENT_SOURCE_CONFLICT", "unknown", rank=1)]
    if view.verdict == "insufficient_evidence":
        return [CandidateCause("INSUFFICIENT_EVIDENCE", "unknown", rank=1)]
    return []


async def _discover(gateway: EvidenceGateway) -> set[str]:
    list_tools = getattr(gateway, "list_tools", None)
    if not callable(list_tools):
        return set()
    return {str(name) for name in await list_tools()}


def _consume(trace: TraceWriter, case_id: str, tool_name: str, payload: dict[str, Any]) -> None:
    ref = evidence_ref_of(payload)
    if ref is None:
        return
    domain = payload.get("domain")
    attributes = {"domain": domain} if isinstance(domain, str) else None
    trace.emit(
        case_id=case_id,
        event_type="tool_result_consumed",
        actor=ACTOR,
        tool_name=tool_name,
        evidence_refs=[ref],
        attributes=attributes,
    )


async def investigate_order_shipment(
    case: dict[str, Any],
    entity: EntityResult,
    gateway: EvidenceGateway,
    trace: TraceWriter,
) -> OrderShipmentResult:
    """Investigate items, sellers and the delivery timeline for resolved orders."""
    case_id = str(case.get("case_id") or "")
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor=COORDINATOR,
        target=ACTOR,
        decision_code="INVESTIGATE_SHIPMENT",
    )
    if entity.status != "resolved" or not entity.resolved_order_ids:
        result = OrderShipmentResult(verdict="insufficient_evidence", timeline_complete=False)
        _handoff(trace, case_id, result)
        return result

    session = _Session(case_id, gateway, await _discover(gateway))
    opened_at = first_datetime(case, ("opened_at",))
    cached_orders = getattr(entity, "order_evidence", {})
    if not isinstance(cached_orders, dict):
        cached_orders = {}

    views: list[_OrderView] = []
    refs: list[str] = []
    for order_id in entity.resolved_order_ids:
        order_payload = cached_orders.get(order_id)
        if not isinstance(order_payload, dict):
            order_payload = await session.call(ORDER_TOOL, order_id=order_id)
            if isinstance(order_payload, dict):
                _consume(trace, case_id, ORDER_TOOL, order_payload)
        order_data = order_payload.get("data") if isinstance(order_payload, dict) else None

        item_payload = await session.call(ITEM_TOOL, order_id=order_id)
        if isinstance(item_payload, dict):
            _consume(trace, case_id, ITEM_TOOL, item_payload)
            ref = evidence_ref_of(item_payload)
            if ref:
                refs.append(ref)
        shipment_payload = await session.call(SHIPMENT_TOOL, order_id=order_id)
        if isinstance(shipment_payload, dict):
            _consume(trace, case_id, SHIPMENT_TOOL, shipment_payload)
            ref = evidence_ref_of(shipment_payload)
            if ref:
                refs.append(ref)
        views.append(
            _analyze(
                order_data,
                item_payload.get("data") if isinstance(item_payload, dict) else None,
                shipment_payload.get("data") if isinstance(shipment_payload, dict) else None,
                opened_at=opened_at,
            )
        )

    view = _aggregate(views)
    result = OrderShipmentResult(
        verdict=view.verdict,
        late_seller_ids=view.late_sellers,
        timeline_complete=view.timeline_complete,
        item_ids=view.item_ids,
        seller_ids=view.seller_ids,
        shipment_ids=view.shipment_ids,
        candidate_causes=_causes(view),
        evidence_refs=[
            ref for ref in dict.fromkeys(refs) if evidence_ref_of({"evidence_ref": ref})
        ],
        order_total_brl=view.order_total_brl,
        order_status=view.order_status,
        shipment_status=view.shipment_status,
        conflicts=view.conflicts,
    )
    _handoff(trace, case_id, result)
    return result


def _handoff(trace: TraceWriter, case_id: str, result: OrderShipmentResult) -> None:
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=ACTOR,
        target=COORDINATOR,
        decision_code=f"SHIPMENT_{result.verdict.upper()}",
        evidence_refs=result.evidence_refs[:20] or None,
        attributes={
            "verdict": result.verdict,
            "timeline_complete": result.timeline_complete,
            "late_seller_count": len(result.late_seller_ids),
            "conflict_count": len(result.conflicts),
        },
    )
