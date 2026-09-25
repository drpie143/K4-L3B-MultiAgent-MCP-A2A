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


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Resolve the order, investigate shipment and payment, then verify the output."""
    case_id = str(case.get("case_id") or "")
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
        shipment = await _shipment(case, entity, gateway, trace)
        payment = await _payment(case, entity, gateway, trace)

    # Call get_policy to obtain authoritative policy rules and evidence ref
    policy_version = str(case.get("policy_version") or "EC_POLICY_V2")
    policy_rules: dict[str, Any] = {}
    policy_ref: str | None = None
    if discovered is None or "get_policy" in discovered:
        for attempt in range(1, 4):
            try:
                policy_payload = await gateway.call(
                    "get_policy", case_id=case_id, policy_version=policy_version
                )
                if isinstance(policy_payload, dict):
                    policy_ref = policy_payload.get("evidence_ref")
                    policy_data = policy_payload.get("data", {})
                    if isinstance(policy_data, dict):
                        policy_rules = policy_data.get("rules", {})
                    if policy_ref:
                        trace.emit(
                            case_id=case_id,
                            event_type="tool_result_consumed",
                            actor="coordinator",
                            tool_name="get_policy",
                            evidence_refs=[policy_ref],
                            attributes={"domain": "policy"},
                        )
                    break
            except Exception:
                if attempt < 3:
                    await asyncio.sleep(0.3)

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="coordinator",
        target="verifier",
        decision_code="READY_FOR_VERIFICATION",
        attributes={"entity_status": entity.status},
    )
    output = verify_and_finalize(
        case,
        entity,
        shipment,
        payment,
        policy_rules=policy_rules,
        policy_ref=policy_ref,
    )
    assessment = output["assessment"]
    trace.emit(
        case_id=case_id,
        event_type="policy_decided",
        actor="verifier",
        decision_code=f"POLICY_{str(assessment['primary_issue']).upper()}",
        attributes={
            "primary_issue": assessment["primary_issue"],
            "case_status": assessment["case_status"],
            "policy_version": policy_version,
        },
    )
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
