"""Evidence signals for the primary-issue decision.

Observed MCP data mixes two timelines for one order: rows around the order's own
purchase date and rows around the complaint (``opened_at``). Rows dated well after the
complaint was opened cannot explain it, so they are ignored for issue detection. Each
order version is compared only with the shipping limit that belongs to it.

Pure functions only: no MCP calls, no trace. Inputs are the ``data`` of MCP envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..utils.evidence import as_float, as_naive, parse_datetime, walk_dicts

# Evidence dated later than this after the complaint is treated as out of scope.
FUTURE_TOLERANCE = timedelta(days=14)
# A shipping limit belongs to an order version when it falls in this window after purchase.
LIMIT_WINDOW = timedelta(days=30)
MONEY_EPS = 0.01

FAILED = {"failed", "failure", "rejected", "declined", "error"}
PENDING = {"pending", "processing", "requested", "submitted", "in_progress", "open"}
COMPLETED = {"completed", "refunded", "succeeded", "success", "settled", "confirmed"}
LOGISTICS_ACTORS = {"logistics", "logistics_provider", "carrier", "courier", "platform"}

# Tie-break when several issues are proven and none was claimed.
PRIORITY = (
    "refund_failed",
    "refund_pending",
    "duplicate_charge",
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "payment_mismatch",
    "valid_split_payment",
)
# Issues whose detector relies on explicit, unambiguous evidence.
STRONG = {
    "refund_failed",
    "refund_pending",
    "duplicate_charge",
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
}


@dataclass
class OrderVersion:
    source: str
    status: str | None
    purchase: datetime | None
    carrier: datetime | None
    delivered: datetime | None
    estimated: datetime | None


@dataclass
class Signals:
    detected: set[str] = field(default_factory=set)
    late_seller_ids: list[str] = field(default_factory=list)
    seller_ids: list[str] = field(default_factory=list)
    shipment_verdict: str = "insufficient_evidence"
    timeline_complete: bool = False
    captured_total: float | None = None
    refunded_total: float | None = None
    duplicate_amount: float = 0.0
    status_conflict: tuple[str, str] | None = None  # (selected source, other source)


def _ts(value: Any) -> datetime | None:
    parsed = parse_datetime(value)
    return as_naive(parsed) if parsed is not None else None


def _text(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _in_scope(when: datetime | None, opened: datetime | None) -> bool:
    return when is None or opened is None or when <= opened + FUTURE_TOLERANCE


def _versions(order: Any, history: Any, order_id: str) -> list[OrderVersion]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for source, data in (("order", order), ("customer", history)):
        for node in walk_dicts(data):
            if str(node.get("order_id")) == order_id and (
                "order_status" in node or "order_purchase_timestamp" in node
            ):
                rows.append((source, node))
    versions: list[OrderVersion] = []
    seen: set[tuple[Any, ...]] = set()
    for source, row in rows:
        version = OrderVersion(
            source=source,
            status=_text(row.get("order_status")) or None,
            purchase=_ts(row.get("order_purchase_timestamp")),
            carrier=_ts(row.get("order_delivered_carrier_date")),
            delivered=_ts(row.get("order_delivered_customer_date")),
            estimated=_ts(row.get("order_estimated_delivery_date")),
        )
        key = (version.status, version.purchase, version.carrier, version.delivered)
        if key not in seen:
            seen.add(key)
            versions.append(version)
    return versions


def _limits(items: Any, shipment: Any) -> list[tuple[str | None, datetime]]:
    limits: list[tuple[str | None, datetime]] = []
    for node in [*walk_dicts(items), *walk_dicts(shipment)]:
        when = _ts(node.get("shipping_limit_date") or node.get("shipping_limit_at"))
        if when is None:
            continue
        seller = node.get("seller_id")
        pair = (str(seller) if seller else None, when)
        if pair not in limits:
            limits.append(pair)
    return limits


def _events(data: Any) -> list[dict[str, Any]]:
    return [node for node in walk_dicts(data) if "event_type" in node]


def analyze(
    case: dict[str, Any],
    order_id: str,
    *,
    order: Any = None,
    history: Any = None,
    items: Any = None,
    shipment: Any = None,
    payments: Any = None,
    refunds: Any = None,
) -> Signals:
    opened = _ts(case.get("opened_at"))
    signals = Signals()
    versions = [v for v in _versions(order, history, order_id) if _in_scope(v.purchase, opened)]
    all_versions = _versions(order, history, order_id)
    limits = _limits(items, shipment)
    signals.seller_ids = list(dict.fromkeys(s for s, _ in limits if s))

    # --- order status and version conflicts -------------------------------------------
    statuses = {v.status for v in versions if v.status}
    base = next((v for v in all_versions if v.source == "order"), None)
    scoped = next((v for v in versions if v.status), None)
    if base and scoped and base.status and scoped.status and base.status != scoped.status:
        signals.status_conflict = (scoped.source, base.source)

    # --- shipment timeline per version -----------------------------------------------
    late_sellers: list[str] = []
    logistics_late = False
    delivered_ok = False
    for version in versions:
        own = [
            (seller, limit)
            for seller, limit in limits
            if version.purchase is not None
            and version.purchase <= limit <= version.purchase + LIMIT_WINDOW
        ]
        seller_late_here = False
        if version.carrier is not None:
            for seller, limit in own:
                if version.carrier > limit:
                    seller_late_here = True
                    if seller and seller not in late_sellers:
                        late_sellers.append(seller)
        if version.delivered and version.estimated:
            if version.delivered > version.estimated and not seller_late_here:
                logistics_late = True
            elif version.delivered <= version.estimated:
                delivered_ok = True
    lost = returned = False
    for event in _events(shipment):
        if not _in_scope(_ts(event.get("event_at")), opened):
            continue
        kind = _text(event.get("event_type"))
        actor = _text(event.get("actor"))
        if "lost" in kind:
            lost = True
        elif "return" in kind:
            returned = True
        elif "late" in kind or "delay" in kind:
            if actor == "seller":
                seller = next((s for s in signals.seller_ids), None)
                if seller and seller not in late_sellers:
                    late_sellers.append(seller)
                elif not seller:
                    late_sellers.append("unknown")
            elif actor in LOGISTICS_ACTORS or not actor:
                logistics_late = True
    late_sellers = [s for s in late_sellers if s != "unknown"] or late_sellers
    signals.late_seller_ids = [s for s in late_sellers if s != "unknown"]
    if late_sellers:
        signals.detected.add("late_delivery_seller")
    if logistics_late or lost or returned:
        signals.detected.add("late_delivery_logistics")
    if lost:
        signals.shipment_verdict = "lost"
    elif returned:
        signals.shipment_verdict = "returned"
    elif signals.late_seller_ids:
        signals.shipment_verdict = "seller_delay"
    elif logistics_late:
        signals.shipment_verdict = "logistics_delay"
    elif delivered_ok:
        signals.shipment_verdict = "on_time"
    signals.timeline_complete = (
        any(v.carrier and v.delivered and v.estimated for v in versions)
        and signals.shipment_verdict != "insufficient_evidence"
    )

    # --- payments ---------------------------------------------------------------------
    rows = [n for n in walk_dicts(payments) if "payment_value" in n]
    keys = [
        (
            str(r.get("payment_sequential")),
            _text(r.get("payment_type")),
            str(r.get("payment_installments")),
            as_float(r.get("payment_value")),
        )
        for r in rows
    ]
    for key in set(keys):
        if keys.count(key) > 1 and key[3]:
            signals.duplicate_amount = max(signals.duplicate_amount, float(key[3]))
    if signals.duplicate_amount > 0:
        signals.detected.add("duplicate_charge")

    captures = [
        e
        for e in _events(payments)
        if "captur" in _text(e.get("event_type")) and _text(e.get("status")) not in FAILED
    ]
    captured_all = sum(as_float(e.get("amount_brl")) or 0.0 for e in captures)
    captured_scoped = sum(
        as_float(e.get("amount_brl")) or 0.0
        for e in captures
        if _in_scope(_ts(e.get("event_at")), opened)
    )
    if captures:
        signals.captured_total = round(captured_scoped, 2)
    elif rows:
        signals.captured_total = round(sum(k[3] or 0.0 for k in keys), 2)

    # The payment timeline flags reconciliation problems explicitly.
    for event in _events(payments):
        if "mismatch" in _text(event.get("event_type")) and _in_scope(
            _ts(event.get("event_at")), opened
        ):
            signals.detected.add("payment_mismatch")

    row_total = sum(k[3] or 0.0 for k in keys)
    row_values = [k[3] for k in keys if k[3] is not None]
    if captures and (
        abs(row_total - captured_all) > MONEY_EPS
        or any(
            all(abs((as_float(e.get("amount_brl")) or 0.0) - v) > MONEY_EPS for v in row_values)
            for e in captures
        )
    ):
        signals.detected.add("payment_mismatch")
    sequences = {k[0] for k in keys}
    if len(sequences) >= 2 and len({k[1] for k in keys}) >= 2:
        signals.detected.add("valid_split_payment")

    # --- refunds ----------------------------------------------------------------------
    refunded = 0.0
    for event in _events(refunds):
        if not _in_scope(_ts(event.get("event_at")), opened):
            continue
        status = _text(event.get("status"))
        kind = _text(event.get("event_type"))
        amount = as_float(event.get("amount_brl")) or 0.0
        if status in FAILED or "fail" in kind:
            signals.detected.add("refund_failed")
        elif status in PENDING or "pending" in kind:
            signals.detected.add("refund_pending")
        elif status in COMPLETED or kind in {"refunded", "refund_completed"}:
            refunded += amount
    signals.refunded_total = round(refunded, 2)

    # --- canceled / unavailable orders that were charged -----------------------------
    net_paid = (signals.captured_total or 0.0) - refunded
    if net_paid > MONEY_EPS:
        if "canceled" in statuses or "cancelled" in statuses:
            signals.detected.add("canceled_order_paid")
        if "unavailable" in statuses:
            signals.detected.add("unavailable_order_paid")
    return signals


def choose_primary(claimed: list[str], detected: set[str]) -> tuple[str, float]:
    """Pick the primary issue and a calibrated confidence.

    A claim confirmed by evidence wins. A claim the evidence does not cover yields to a
    strong, explicit signal. Otherwise the claim stands with lower confidence.
    """
    topics = [t for t in claimed if t != "requested_full_refund"]
    for topic in topics:
        if topic in detected:
            return topic, 0.92
    if "unsupported_claim" in topics:
        return "unsupported_claim", 0.75
    strong = [issue for issue in PRIORITY if issue in detected and issue in STRONG]
    if strong:
        return strong[0], 0.6
    if topics:
        return topics[0], 0.55
    fallback = next((issue for issue in PRIORITY if issue in detected), None)
    return (fallback, 0.5) if fallback else ("unsupported_claim", 0.5)
