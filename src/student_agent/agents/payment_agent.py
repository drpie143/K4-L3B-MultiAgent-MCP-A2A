from __future__ import annotations

from typing import Any

from ..mcp_gateway import EvidenceGateway
from ..models.messages import (
    CandidateCause,
    EntityResult,
    FinancialResolution,
    PaymentResult,
    PaymentVerdict,
    RefundLine,
)
from ..trace import TraceWriter


async def investigate_payment(
    case: dict[str, Any],
    entity: EntityResult,
    gateway: EvidenceGateway,
    trace: TraceWriter,
) -> PaymentResult:
    """Investigate payment transactions, reconciliations, duplicates, and refunds.

    Strictly satisfies L3B contracts and scoring rules:
    - Safe fallback if entity is unresolved (no unnecessary MCP calls, zero penalty).
    - Uses tool discovery to invoke payment and refund tools.
    - Emits observable trace events: tool_result_consumed and handoff.
    - Calibrates financial resolution and candidate causes.
    """
    case_id: str = case.get("case_id", "")

    # 1. Fallback if entity is not resolved or missing order ID
    if entity.status in ("not_found", "ambiguous") or not entity.resolved_order_ids:
        # Do not call MCP tools; preserve budget and avoid speculative data
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="payment-agent",
            target="coordinator",
            attributes={"status": "unresolved_entity_fallback"},
        )
        return PaymentResult(
            verdict="insufficient_evidence",
            captured_total_brl=None,
            refunded_total_brl=None,
            refundable_total_brl=None,
            payment_references=[],
            financial_resolution=FinancialResolution(
                currency="BRL",
                recommended_refund_brl=0.0,
                refund_lines=[],
            ),
            candidate_causes=[
                CandidateCause(
                    cause_code="INSUFFICIENT_EVIDENCE",
                    party_type="unknown",
                    rank=1,
                )
            ],
            evidence_refs=[],
        )

    order_id = entity.resolved_order_ids[0]
    discovered_tools = await gateway.list_tools()
    evidence_refs: list[str] = []

    # 2. Query Payment details / transactions
    payment_data: list[dict[str, Any]] = []
    payment_tool_name = _find_matching_tool(
        discovered_tools,
        candidates=[
            "get_payment_details",
            "get_order_payments",
            "get_payment_transactions",
            "get_payments",
        ],
    )

    if payment_tool_name:
        try:
            ev_payment = await gateway.call(payment_tool_name, case_id=case_id, order_id=order_id)
            ev_ref = ev_payment.get("evidence_ref")
            if ev_ref:
                evidence_refs.append(ev_ref)
                trace.emit(
                    case_id=case_id,
                    event_type="tool_result_consumed",
                    actor="payment-agent",
                    tool_name=payment_tool_name,
                    evidence_refs=[ev_ref],
                )
            raw_content = ev_payment.get("data", {})
            if isinstance(raw_content, list):
                payment_data = raw_content
            elif isinstance(raw_content, dict):
                payment_data = (
                    raw_content.get("payments") or raw_content.get("transactions") or [raw_content]
                )
        except Exception:
            payment_data = []

    # 3. Query Refund details (if refund-specific tool exists)
    refund_data: list[dict[str, Any]] = []
    refund_tool_name = _find_matching_tool(
        discovered_tools,
        candidates=["get_refund_status", "get_order_refunds", "get_refunds"],
    )

    if refund_tool_name:
        try:
            ev_refund = await gateway.call(refund_tool_name, case_id=case_id, order_id=order_id)
            ev_ref = ev_refund.get("evidence_ref")
            if ev_ref:
                evidence_refs.append(ev_ref)
                trace.emit(
                    case_id=case_id,
                    event_type="tool_result_consumed",
                    actor="payment-agent",
                    tool_name=refund_tool_name,
                    evidence_refs=[ev_ref],
                )
            raw_content = ev_refund.get("data", {})
            if isinstance(raw_content, list):
                refund_data = raw_content
            elif isinstance(raw_content, dict):
                refund_data = raw_content.get("refunds") or [raw_content]
        except Exception:
            refund_data = []

    # 4. If no payment data retrieved, return insufficient_evidence
    if not payment_data and not refund_data:
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor="payment-agent",
            target="coordinator",
            attributes={"status": "no_payment_data"},
        )
        return PaymentResult(
            verdict="insufficient_evidence",
            captured_total_brl=None,
            refunded_total_brl=None,
            refundable_total_brl=None,
            payment_references=[],
            financial_resolution=FinancialResolution(
                currency="BRL",
                recommended_refund_brl=0.0,
                refund_lines=[],
            ),
            candidate_causes=[
                CandidateCause(
                    cause_code="INSUFFICIENT_EVIDENCE",
                    party_type="unknown",
                    rank=1,
                )
            ],
            evidence_refs=evidence_refs,
        )

    # 5. Extract payment references and calculate totals
    payment_references: list[str] = []
    captured_total = 0.0
    refunded_total = 0.0

    # Collect transaction amounts and detect duplicates
    seen_transactions: dict[tuple[float, str], int] = {}
    duplicate_amount = 0.0
    has_duplicate = False

    for item in payment_data:
        ref = (
            item.get("payment_reference")
            or item.get("payment_id")
            or item.get("transaction_id")
            or item.get("payment_sequential")
        )
        if ref:
            payment_references.append(str(ref))

        value = float(item.get("payment_value", 0.0))
        method = str(item.get("payment_type", "unknown"))
        tx_status = str(item.get("status", "captured")).lower()

        if tx_status in ("captured", "authorized", "completed", "success"):
            captured_total += value
            # Check for duplicate charge (same value & method and flagged or identical)
            key = (value, method)
            seen_transactions[key] = seen_transactions.get(key, 0) + 1
            if seen_transactions[key] > 1 or item.get("is_duplicate"):
                has_duplicate = True
                duplicate_amount = max(duplicate_amount, value)

    # Calculate refunded totals
    has_failed_refund = False
    has_pending_refund = False
    pending_refund_amount = 0.0

    for item in refund_data:
        r_status = str(item.get("refund_status") or item.get("status", "")).lower()
        r_amount = float(item.get("amount") or item.get("refund_amount", 0.0))
        if r_status in ("completed", "refunded", "success"):
            refunded_total += r_amount
        elif r_status in ("failed", "rejected", "error"):
            has_failed_refund = True
            pending_refund_amount = max(pending_refund_amount, r_amount)
        elif r_status in ("pending", "processing", "in_review"):
            has_pending_refund = True
            pending_refund_amount = max(pending_refund_amount, r_amount)

    refundable_total = max(0.0, captured_total - refunded_total)

    # 6. Determine Payment Verdict
    verdict: PaymentVerdict
    candidate_causes: list[CandidateCause] = []
    recommended_refund = 0.0
    refund_lines: list[RefundLine] = []

    if has_failed_refund:
        verdict = "refund_failed"
        recommended_refund = (
            pending_refund_amount if pending_refund_amount > 0 else refundable_total
        )
        candidate_causes.append(
            CandidateCause(
                cause_code="REFUND_GATEWAY_FAILURE", party_type="payment_provider", rank=1
            )
        )
        refund_lines.append(
            RefundLine(
                reason_code="REFUND_RETRY", amount_brl=recommended_refund, entity_id=order_id
            )
        )
    elif has_pending_refund:
        verdict = "refund_pending"
        recommended_refund = (
            pending_refund_amount if pending_refund_amount > 0 else refundable_total
        )
        candidate_causes.append(
            CandidateCause(
                cause_code="REFUND_PENDING_SETTLEMENT", party_type="payment_provider", rank=1
            )
        )
        refund_lines.append(
            RefundLine(
                reason_code="REFUND_PENDING", amount_brl=recommended_refund, entity_id=order_id
            )
        )
    elif has_duplicate:
        verdict = "duplicate_capture"
        recommended_refund = duplicate_amount
        candidate_causes.append(
            CandidateCause(cause_code="DUPLICATE_CHARGE", party_type="payment_provider", rank=1)
        )
        refund_lines.append(
            RefundLine(
                reason_code="DUPLICATE_CHARGE_REFUND",
                amount_brl=recommended_refund,
                entity_id=order_id,
            )
        )
    elif refunded_total > 0 and refundable_total < 0.01:
        verdict = "refunded"
        recommended_refund = 0.0
    else:
        # Check expected amount if available in case
        expected_total = case.get("expected_total_brl") or case.get("order_total_brl")
        if expected_total is not None and abs(captured_total - float(expected_total)) > 1.0:
            verdict = "capture_mismatch"
            diff = abs(captured_total - float(expected_total))
            candidate_causes.append(
                CandidateCause(cause_code="PAYMENT_AMOUNT_MISMATCH", party_type="platform", rank=1)
            )
            if captured_total > float(expected_total):
                recommended_refund = diff
                refund_lines.append(
                    RefundLine(reason_code="OVERCHARGE_REFUND", amount_brl=diff, entity_id=order_id)
                )
        else:
            verdict = "reconciled"
            recommended_refund = 0.0

    financial_resolution = FinancialResolution(
        currency="BRL",
        recommended_refund_brl=recommended_refund,
        refund_lines=refund_lines,
    )

    # 7. Trace handoff to coordinator
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor="payment-agent",
        target="coordinator",
        decision_code=f"PAYMENT_{verdict.upper()}",
        attributes={
            "verdict": verdict,
            "captured_total_brl": round(captured_total, 2),
            "refunded_total_brl": round(refunded_total, 2),
            "recommended_refund_brl": round(recommended_refund, 2),
        },
    )

    return PaymentResult(
        verdict=verdict,
        captured_total_brl=round(captured_total, 2),
        refunded_total_brl=round(refunded_total, 2),
        refundable_total_brl=round(refundable_total, 2),
        payment_references=payment_references,
        financial_resolution=financial_resolution,
        candidate_causes=candidate_causes,
        evidence_refs=evidence_refs,
    )


def _find_matching_tool(available_tools: list[str], candidates: list[str]) -> str | None:
    """Find matching MCP tool name dynamically from available tools."""
    tool_set = set(available_tools)
    for candidate in candidates:
        if candidate in tool_set:
            return candidate
    for tool in available_tools:
        for candidate in candidates:
            if candidate in tool or tool in candidate:
                return tool
    return None
