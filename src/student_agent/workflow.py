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
from .utils.evidence import evidence_ref_of, valid_evidence_refs
from .utils.evidence_store import StoredGateway, store_root

POLICY_TOOL = "get_policy"


async def _tools(gateway: EvidenceGateway) -> set[str] | None:
    list_tools = getattr(gateway, "list_tools", None)
    if not callable(list_tools):
        return None
    return {str(name) for name in await list_tools()}


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


async def _policy(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter, tools: set[str] | None
) -> dict[str, Any] | None:
    """Fetch the case's policy version once; the verifier's rules follow that policy."""
    case_id = str(case.get("case_id") or "")
    version = case.get("policy_version")
    if not isinstance(version, str) or not version:
        return None
    if tools is not None and POLICY_TOOL not in tools:
        return None
    try:
        payload = await gateway.call(POLICY_TOOL, case_id=case_id, policy_version=version)
    except Exception:
        return None
    ref = evidence_ref_of(payload)
    if ref is None:
        return None
    trace.emit(
        case_id=case_id,
        event_type="tool_result_consumed",
        actor="coordinator",
        tool_name=POLICY_TOOL,
        evidence_refs=[ref],
        attributes={"domain": payload.get("domain"), "policy_version": version},
    )
    return payload


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Resolve the order, investigate shipment and payment, then verify the output."""
    case_id = str(case.get("case_id") or "")
    root = store_root()
    if root is not None and isinstance(gateway, EvidenceGateway):
        gateway = StoredGateway(gateway, root)  # type: ignore[assignment]
    discovered = await _tools(gateway)
    entity = await resolve_entity(
        case,
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
            _shipment(case, entity, gateway, trace),
            _payment(case, entity, gateway, trace),
        )
    policy = await _policy(case, gateway, trace, discovered)

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="coordinator",
        target="verifier",
        decision_code="READY_FOR_VERIFICATION",
        attributes={"entity_status": entity.status},
    )
    policy_data = policy.get("data") if isinstance(policy, dict) else None
    output = verify_and_finalize(
        case, entity, shipment, payment, policy_data if isinstance(policy_data, dict) else None
    )
    policy_ref = evidence_ref_of(policy)
    if policy_ref is not None:
        output["evidence_refs"] = valid_evidence_refs([*output["evidence_refs"], policy_ref])
        trace.emit(
            case_id=case_id,
            event_type="policy_decided",
            actor="coordinator",
            decision_code=f"POLICY_{output['assessment']['case_status'].upper()}",
            evidence_refs=[policy_ref],
            attributes={"policy_version": str(case.get("policy_version"))},
        )
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
