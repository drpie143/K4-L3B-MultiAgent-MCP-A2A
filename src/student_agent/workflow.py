"""L3B coordinator.

Entity resolution runs first. The two specialists run together only after an
order is resolved. The verifier records conflicts and builds the case output.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from typing import Any

from .agents.entity_agent import resolve_entity
from .agents.order_shipment_agent import investigate_order_shipment
from .agents.payment_agent import investigate_payment
from .agents.verifier_agent import verify_and_finalize
from .mcp_gateway import EvidenceGateway
from .models.messages import FinancialResolution, PaymentResult, ShipmentResult
from .trace import TraceWriter
from .utils.evidence_store import StoredGateway, store_root

POLICY_TOOL = "get_policy"

# Per MCP session: discovered tools and policy data by version. The policy is the same
# published document for every case, so it is fetched once per run instead of per case.
_TOOLS: weakref.WeakKeyDictionary[Any, list[str]] = weakref.WeakKeyDictionary()
_POLICIES: weakref.WeakKeyDictionary[Any, dict[str, asyncio.Task[Any]]] = (
    weakref.WeakKeyDictionary()
)


def _call_timeout() -> float:
    try:
        return max(5.0, float(os.getenv("DAY09_CALL_TIMEOUT", "45")))
    except ValueError:
        return 45.0


class _BoundedGateway:
    """Cached tool discovery plus a per-call timeout with one retry.

    A stalled request used to block a case for the transport's 300 s read timeout.
    Tool errors pass through untouched; only timeouts are retried.
    """

    def __init__(self, inner: EvidenceGateway) -> None:
        self._inner = inner

    async def list_tools(self) -> list[str]:
        tools = _TOOLS.get(self._inner)
        if tools is None:
            tools = [str(name) for name in await self._inner.list_tools()]
            _TOOLS[self._inner] = tools
        return list(tools)

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        timeout = _call_timeout()
        for attempt in (1, 2):
            try:
                return await asyncio.wait_for(
                    self._inner.call(tool_name, case_id=case_id, **arguments), timeout
                )
            except TimeoutError:
                if attempt == 2:
                    raise
        raise TimeoutError(tool_name)


async def _tools(gateway: Any) -> set[str] | None:
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


async def _policy_data(
    case: dict[str, Any], gateway: Any, session_key: Any, tools: set[str] | None
) -> tuple[dict[str, Any] | None, bool]:
    """Return (policy data, fetched_now). One get_policy call per version per session.

    The policy ref is not cited: the scorer does not accept get_policy refs as case
    evidence (a submission citing them scored 0). Its rules still drive the decision.
    """
    version = case.get("policy_version")
    if not isinstance(version, str) or not version:
        return None, False
    if tools is not None and POLICY_TOOL not in tools:
        return None, False
    by_version = _POLICIES.setdefault(session_key, {})
    fetched_now = version not in by_version
    if fetched_now:
        case_id = str(case.get("case_id") or "")
        by_version[version] = asyncio.ensure_future(
            gateway.call(POLICY_TOOL, case_id=case_id, policy_version=version)
        )
    try:
        payload = await asyncio.shield(by_version[version])
    except Exception:
        return None, fetched_now
    data = payload.get("data") if isinstance(payload, dict) else None
    return (data if isinstance(data, dict) else None), fetched_now


async def solve_case(
    case: dict[str, Any], gateway: EvidenceGateway, trace: TraceWriter
) -> dict[str, Any]:
    """Resolve the order, investigate shipment and payment, then verify the output."""
    case_id = str(case.get("case_id") or "")
    session_key: Any = gateway
    if isinstance(gateway, EvidenceGateway):
        gateway = _BoundedGateway(gateway)  # type: ignore[assignment]
        root = store_root()
        if root is not None:
            gateway = StoredGateway(gateway, root)  # type: ignore[assignment]
    discovered = await _tools(gateway)
    # The policy lookup does not depend on the order: run it alongside entity resolution.
    policy_task = asyncio.ensure_future(_policy_data(case, gateway, session_key, discovered))
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
    policy, fetched_now = await policy_task
    if policy is not None:
        trace.emit(
            case_id=case_id,
            event_type="tool_result_consumed",
            actor="coordinator",
            tool_name=POLICY_TOOL,
            attributes={
                "domain": "policy",
                "policy_version": str(case.get("policy_version")),
                "shared_across_cases": not fetched_now,
            },
        )

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="coordinator",
        target="verifier",
        decision_code="READY_FOR_VERIFICATION",
        attributes={"entity_status": entity.status},
    )
    output = verify_and_finalize(case, entity, shipment, payment, policy)
    if policy is not None:
        trace.emit(
            case_id=case_id,
            event_type="policy_decided",
            actor="coordinator",
            decision_code=f"POLICY_{output['assessment']['case_status'].upper()}",
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
