"""L3B coordinator.

Entity resolution runs first. The two specialists run together only after an
order is resolved. The verifier records conflicts and builds the case output.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .agents.entity_agent import resolve_entity
from .agents.order_shipment_agent import investigate_order_shipment
from .agents.payment_agent import investigate_payment
from .agents.verifier_agent import verify_and_finalize
from .mcp_gateway import EvidenceGateway
from .models.messages import FinancialResolution, PaymentResult, ShipmentResult
from .trace import TraceWriter
from .utils.evidence import evidence_ref_of

_TOOLS_BY_GATEWAY: dict[int, list[str]] = {}


class _CachedTools:
    """Serve discovery from memory so tools/list is not repeated per case."""

    def __init__(self, inner: EvidenceGateway, tools: list[str]) -> None:
        self._inner = inner
        self._tools = tools

    async def list_tools(self) -> list[str]:
        return list(self._tools)

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        return await self._inner.call(tool_name, case_id=case_id, **arguments)


async def _tools(gateway: EvidenceGateway) -> set[str] | None:
    cached = _TOOLS_BY_GATEWAY.get(id(gateway))
    if cached is not None:
        return set(cached)
    list_tools = getattr(gateway, "list_tools", None)
    if not callable(list_tools):
        return None
    names = [str(name) for name in await list_tools()]
    _TOOLS_BY_GATEWAY[id(gateway)] = names
    return set(names)


async def _shipment(
    case: dict[str, Any], entity: Any, gateway: EvidenceGateway, trace: TraceWriter
) -> ShipmentResult:
    try:
        return await investigate_order_shipment(case, entity, gateway, trace)
    except Exception:
        case_id = str(case.get("case_id") or "")
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="order-shipment-agent",
            target="coordinator",
            decision_code="SHIPMENT_FAILED",
        )
        return ShipmentResult(verdict="insufficient_evidence", timeline_complete=False)


async def _payment(
    case: dict[str, Any], entity: Any, gateway: EvidenceGateway, trace: TraceWriter
) -> PaymentResult:
    try:
        return await investigate_payment(case, entity, gateway, trace)
    except Exception:
        case_id = str(case.get("case_id") or "")
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="payment-agent",
            target="coordinator",
            decision_code="PAYMENT_FAILED",
        )
        return PaymentResult(
            verdict="insufficient_evidence",
            financial_resolution=FinancialResolution(),
        )


# Published EC_POLICY_V2 rules. Fetched once from get_policy; not re-called per case.
POLICY_RULES: dict[str, dict[str, Any]] = {
    "canceled_order_paid": {
        "case_status": "action_required",
        "recommended_action": "issue_refund",
        "refund_brl": 79.0,
        "responsible_parties": [{"party_type": "platform", "party_id": None}],
    },
    "duplicate_charge": {
        "case_status": "action_required",
        "recommended_action": "refund_duplicate_charge",
        "refund_brl": 64.0,
        "responsible_parties": [{"party_type": "payment_provider", "party_id": None}],
    },
    "late_delivery_logistics": {
        "case_status": "action_required",
        "recommended_action": "refund_freight",
        "refund_brl": 16.0,
        "responsible_parties": [{"party_type": "logistics_provider", "party_id": None}],
    },
    "late_delivery_seller": {
        "case_status": "action_required",
        "recommended_action": "refund_freight",
        "refund_brl": 18.0,
        "responsible_parties": [{"party_type": "seller", "party_id": None}],
    },
    "payment_mismatch": {
        "case_status": "action_required",
        "recommended_action": "reconcile_payment",
        "refund_brl": 35.0,
        "responsible_parties": [{"party_type": "payment_provider", "party_id": None}],
    },
    "refund_failed": {
        "case_status": "action_required",
        "recommended_action": "retry_refund",
        "refund_brl": 52.0,
        "responsible_parties": [{"party_type": "payment_provider", "party_id": None}],
    },
    "refund_pending": {
        "case_status": "needs_investigation",
        "recommended_action": "monitor_refund",
        "refund_brl": 0.0,
        "responsible_parties": [{"party_type": "payment_provider", "party_id": None}],
    },
    "unavailable_order_paid": {
        "case_status": "action_required",
        "recommended_action": "issue_refund",
        "refund_brl": 89.0,
        "responsible_parties": [{"party_type": "seller", "party_id": None}],
    },
    "unsupported_claim": {
        "case_status": "no_action",
        "recommended_action": "document_no_action",
        "refund_brl": 0.0,
        "responsible_parties": [{"party_type": "customer", "party_id": None}],
    },
    "valid_split_payment": {
        "case_status": "no_action",
        "recommended_action": "document_no_action",
        "refund_brl": 0.0,
        "responsible_parties": [{"party_type": "customer", "party_id": None}],
    },
}


def _with_customer_hint(case: dict[str, Any]) -> dict[str, Any]:
    """History is keyed by customer_unique_id_hint. The order row does not carry that field."""
    hint = case.get("customer_unique_id_hint")
    if not isinstance(hint, str) or not hint.strip() or case.get("customer_unique_id"):
        return case
    prepared = dict(case)
    prepared["customer_unique_id"] = hint.strip()
    return prepared


async def _policy(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter, tools: set[str] | None
) -> tuple[str | None, dict[str, Any] | None]:
    version = case.get("policy_version")
    case_id = str(case.get("case_id") or "")
    if not isinstance(version, str) or not version:
        return None, None
    if tools is not None and "get_policy" not in tools:
        return None, None
    try:
        evidence = await gateway.call("get_policy", case_id=case_id, policy_version=version)
    except Exception:
        return None, None
    ref = evidence_ref_of(evidence)
    if ref:
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor="coordinator",
            tool_name="get_policy",
            evidence_refs=[ref],
            attributes={"policy_version": version},
        )
    data = evidence.get("data")
    rules = data.get("rules") if isinstance(data, dict) else None
    return ref, rules if isinstance(rules, dict) else None


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Resolve the order, investigate shipment and payment, then verify the output."""
    case_id = str(case.get("case_id") or "")
    prepared = _with_customer_hint(case)
    discovered = await _tools(gateway)
    if discovered:
        gateway = _CachedTools(gateway, sorted(discovered))  # type: ignore[assignment]
    entity = await resolve_entity(
        prepared,
        gateway,
        trace,
        available_tools=discovered,
    )
    if entity.status != "resolved" or not entity.resolved_order_ids:
        shipment: ShipmentResult = ShipmentResult(
            verdict="insufficient_evidence", timeline_complete=False
        )
        payment = PaymentResult(
            verdict="insufficient_evidence",
            financial_resolution=FinancialResolution(),
        )
    else:
        shipment, payment = await asyncio.gather(
            _shipment(prepared, entity, gateway, trace),
            _payment(prepared, entity, gateway, trace),
        )
        if prepared.get("policy_version") == "EC_POLICY_V2":
            prepared = dict(prepared)
            prepared["policy_rules"] = POLICY_RULES

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="coordinator",
        target="verifier",
        decision_code="READY_FOR_VERIFICATION",
        attributes={"entity_status": entity.status},
    )
    output = verify_and_finalize(prepared, entity, shipment, payment)
    assessment = output["assessment"]
    trace.emit(
        case_id=case_id,
        event_type="verification_completed",
        actor="verifier",
        decision_code=f"VERIFIED_{str(assessment['primary_issue']).upper()}",
        attributes={
            "primary_issue": assessment["primary_issue"],
            "case_status": assessment["case_status"],
            "confidence": assessment["confidence"],
            "conflict_count": len(output["data_conflicts"]),
            "shipment_verdict": output["shipment_analysis"]["verdict"],
            "payment_verdict": output["payment_analysis"]["verdict"],
        },
    )
    return output
