"""Payment, refund and financial specialist.

Discovers payment and refund tools, then returns totals, a verdict and candidate
causes. A captured amount is never treated as a refund by itself.
"""

from __future__ import annotations

from typing import Any

from ..cases import CASE_ID_PATTERN
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
from ..utils.evidence import as_float, evidence_ref_of, walk_dicts

ACTOR = "payment-agent"
COORDINATOR = "coordinator"

PAYMENT_TOOLS = (
    "get_payment_timeline",
    "get_order_payments",
    "get_payment_details",
    "get_payment_transactions",
    "get_payments",
)
REFUND_TOOLS = (
    "get_refund_timeline",
    "get_refund_status",
    "get_order_refunds",
    "get_refunds",
)
CAPTURED_STATUSES = {"captured", "authorized", "completed", "success", "succeeded", "paid"}
FAILED_TOKENS = {"failed", "failure", "rejected", "declined", "error"}
PENDING_TOKENS = {"pending", "processing", "review", "requested", "submitted"}
COMPLETED_TOKENS = {"completed", "refunded", "success", "succeeded"}


def _claim_needs_refund(case: dict[str, Any]) -> bool:
    request = case.get("customer_request")
    claims = request.get("claims") if isinstance(request, dict) else None
    if not isinstance(claims, list):
        return True
    topics = {claim.get("topic") for claim in claims if isinstance(claim, dict)}
    return bool(topics & {"refund_failed", "refund_pending"})


def _captured_by_day(payloads: list[Any]) -> float | None:
    """Largest single-day sum of capture events. Ignores the other timeline's total."""
    clusters: dict[str, float] = {}
    for payload in payloads:
        for node in walk_dicts(payload):
            if str(node.get("event_type", "")).lower() != "captured":
                continue
            amount = as_float(node.get("amount_brl"))
            if amount is None:
                continue
            day = str(node.get("event_at") or "")[:10]
            clusters[day] = clusters.get(day, 0.0) + amount
    if not clusters:
        return None
    return round(max(clusters.values()), 2)


def _find_matching_tool(
    available_tools: list[str], candidates: tuple[str, ...] | list[str]
) -> str | None:
    tool_set = set(available_tools)
    for candidate in candidates:
        if candidate in tool_set:
            return candidate
    for tool in available_tools:
        for candidate in candidates:
            if candidate in tool or tool in candidate:
                return tool
    return None


def _payment_rows(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in walk_dicts(data):
        if "payment_value" not in node or "refund_amount" in node or "refund_status" in node:
            continue
        identity = (
            node.get("payment_id")
            or node.get("payment_reference")
            or node.get("transaction_id")
        )
        if identity is not None:
            key = str(identity)
            if key in seen:
                continue
            seen.add(key)
        rows.append(node)
    return rows


def _refund_rows(data: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in walk_dicts(data):
        if any(key in node for key in ("refund_status", "refund_amount", "refund_id")):
            rows.append(node)
    return rows


def _bucket(status: str) -> str:
    parts = set(status.lower().replace("-", "_").split("_"))
    text = status.lower()
    if parts & FAILED_TOKENS or "fail" in text:
        return "failed"
    if parts & PENDING_TOKENS or "pending" in text:
        return "pending"
    if parts & COMPLETED_TOKENS:
        return "completed"
    return ""


def _summarize_refunds(rows: list[dict[str, Any]]) -> tuple[float, float, float, bool, bool]:
    """Return completed total, failed amount, pending amount, and the two flags."""
    groups: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        identity = row.get("refund_id") or row.get("id") or f"row-{index}"
        group = groups.setdefault(str(identity), {"buckets": set(), "amount": 0.0})
        status = str(row.get("refund_status") or row.get("status") or row.get("event_type") or "")
        bucket = _bucket(status)
        if bucket:
            group["buckets"].add(bucket)
        amount = as_float(row.get("refund_amount"))
        if amount is None:
            amount = as_float(row.get("amount"))
        if amount is None:
            amount = as_float(row.get("amount_brl"))
        if amount is not None:
            group["amount"] = max(float(group["amount"]), amount)

    refunded = failed_amount = pending_amount = 0.0
    has_failed = has_pending = False
    for group in groups.values():
        buckets = group["buckets"]
        amount = float(group["amount"])
        if "failed" in buckets and "completed" not in buckets:
            has_failed = True
            failed_amount = max(failed_amount, amount)
        elif "pending" in buckets and "completed" not in buckets:
            has_pending = True
            pending_amount = max(pending_amount, amount)
        elif "completed" in buckets:
            refunded += amount
    return refunded, failed_amount, pending_amount, has_failed, has_pending


def _duplicate_amount(rows: list[dict[str, Any]], payload: Any) -> float:
    """Return the extra captured amount, or 0 when the rows are a valid split."""
    amount = 0.0
    explicit = False
    for node in walk_dicts(payload):
        label = " ".join(
            str(node.get(key, "")) for key in ("event_type", "type", "status")
        ).lower()
        if node.get("is_duplicate") or "duplicate" in label:
            explicit = True
            flagged = as_float(node.get("payment_value"))
            if flagged is None:
                flagged = as_float(node.get("amount"))
            if flagged is not None:
                amount = max(amount, flagged)
    sequentials: dict[str, list[float]] = {}
    unlabeled: list[tuple[float, str]] = []
    for row in rows:
        value = as_float(row.get("payment_value")) or 0.0
        sequential = row.get("payment_sequential")
        if sequential is not None:
            sequentials.setdefault(str(sequential), []).append(value)
        else:
            unlabeled.append((value, str(row.get("payment_type", "unknown"))))
    for values in sequentials.values():
        if len(values) > 1:
            explicit = True
            amount = max(amount, max(values))
    counts: dict[tuple[float, str], int] = {}
    for key in unlabeled:
        counts[key] = counts.get(key, 0) + 1
    for (value, _), count in counts.items():
        if count > 1:
            explicit = True
            amount = max(amount, value)
    return amount if explicit else 0.0


def _references(rows: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for row in rows:
        ref = (
            row.get("payment_reference")
            or row.get("payment_id")
            or row.get("transaction_id")
            or row.get("payment_sequential")
        )
        if ref is not None and str(ref) not in refs:
            refs.append(str(ref))
    return refs


async def _call(
    gateway: EvidenceGateway, tool_name: str, case_id: str, order_id: str
) -> dict[str, Any] | None:
    try:
        payload = await gateway.call(tool_name, case_id=case_id, order_id=order_id)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _emit_consumed(
    trace: TraceWriter, case_id: str, tool_name: str, payload: dict[str, Any]
) -> str | None:
    ref = evidence_ref_of(payload)
    if ref is None:
        return None
    trace.emit(
        case_id=case_id,
        event_type="tool_result_consumed",
        actor=ACTOR,
        tool_name=tool_name,
        evidence_refs=[ref],
    )
    return ref


async def investigate_payment(
    case: dict[str, Any],
    entity: EntityResult,
    gateway: EvidenceGateway,
    trace: TraceWriter,
) -> PaymentResult:
    """Investigate captures, splits, duplicates and refunds for the resolved orders."""
    case_id = case.get("case_id", "")
    if isinstance(case_id, str) and CASE_ID_PATTERN.fullmatch(case_id):
        trace.emit(
            case_id=case_id,
            event_type="task_assigned",
            actor=COORDINATOR,
            target=ACTOR,
            decision_code="INVESTIGATE_PAYMENT",
        )

    if entity.status in {"not_found", "ambiguous"} or not entity.resolved_order_ids:
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=ACTOR,
            target=COORDINATOR,
            decision_code="PAYMENT_INSUFFICIENT_EVIDENCE",
            attributes={"status": "unresolved_entity_fallback"},
        )
        return PaymentResult(
            verdict="insufficient_evidence",
            candidate_causes=[CandidateCause("INSUFFICIENT_EVIDENCE", "unknown", rank=1)],
            financial_resolution=FinancialResolution(),
        )

    discovered = await gateway.list_tools()
    payment_tool = _find_matching_tool(discovered, PAYMENT_TOOLS)
    refund_tool = _find_matching_tool(discovered, REFUND_TOOLS)
    if refund_tool and not _claim_needs_refund(case):
        refund_tool = None
    fallback_payment = None
    if payment_tool == "get_payment_timeline" and "get_order_payments" in set(discovered):
        fallback_payment = "get_order_payments"

    rows: list[dict[str, Any]] = []
    refund_rows: list[dict[str, Any]] = []
    evidence_refs: list[str] = []
    raw_payloads: list[Any] = []
    for order_id in entity.resolved_order_ids:
        if payment_tool:
            payload = await _call(gateway, payment_tool, case_id, order_id)
            if payload is not None:
                raw_payloads.append(payload.get("data"))
                ref = _emit_consumed(trace, case_id, payment_tool, payload)
                if ref:
                    evidence_refs.append(ref)
                found = _payment_rows(payload.get("data"))
                if not found and fallback_payment:
                    extra = await _call(gateway, fallback_payment, case_id, order_id)
                    if extra is not None:
                        raw_payloads.append(extra.get("data"))
                        ref = _emit_consumed(trace, case_id, fallback_payment, extra)
                        if ref:
                            evidence_refs.append(ref)
                        found = _payment_rows(extra.get("data"))
                rows.extend(found)
        if refund_tool:
            payload = await _call(gateway, refund_tool, case_id, order_id)
            if payload is not None:
                ref = _emit_consumed(trace, case_id, refund_tool, payload)
                if ref:
                    evidence_refs.append(ref)
                refund_rows.extend(_refund_rows(payload.get("data")))

    if not rows and not refund_rows:
        trace.emit(
            case_id=case_id,
            event_type="handoff",
            actor=ACTOR,
            target=COORDINATOR,
            decision_code="PAYMENT_INSUFFICIENT_EVIDENCE",
            attributes={"status": "no_payment_data"},
        )
        return PaymentResult(
            verdict="insufficient_evidence",
            candidate_causes=[CandidateCause("INSUFFICIENT_EVIDENCE", "unknown", rank=1)],
            financial_resolution=FinancialResolution(),
            evidence_refs=evidence_refs,
        )

    captured_rows: list[dict[str, Any]] = []
    captured_total = 0.0
    for row in rows:
        value = as_float(row.get("payment_value")) or 0.0
        status = str(row.get("status", "captured")).lower()
        if status in CAPTURED_STATUSES:
            captured_total += value
            captured_rows.append(row)

    refunded_total, failed_amount, pending_amount, has_failed, has_pending = _summarize_refunds(
        refund_rows
    )
    clustered = _captured_by_day(raw_payloads)
    if clustered is not None:
        captured_total = clustered
    refundable_total = max(0.0, captured_total - refunded_total)
    duplicate_amount = _duplicate_amount(captured_rows, raw_payloads)
    order_id = entity.resolved_order_ids[0]

    verdict: PaymentVerdict
    causes: list[CandidateCause] = []
    recommended = 0.0
    lines: list[RefundLine] = []
    if has_failed:
        verdict = "refund_failed"
        recommended = failed_amount if failed_amount > 0 else refundable_total
        causes.append(CandidateCause("REFUND_GATEWAY_FAILURE", "payment_provider", rank=1))
        lines.append(RefundLine("REFUND_RETRY", recommended, order_id))
    elif has_pending:
        verdict = "refund_pending"
        recommended = pending_amount if pending_amount > 0 else refundable_total
        causes.append(CandidateCause("REFUND_PENDING_SETTLEMENT", "payment_provider", rank=1))
        lines.append(RefundLine("REFUND_PENDING", recommended, order_id))
    elif duplicate_amount > 0:
        verdict = "duplicate_capture"
        recommended = duplicate_amount
        causes.append(CandidateCause("DUPLICATE_CHARGE", "payment_provider", rank=1))
        lines.append(RefundLine("DUPLICATE_CHARGE_REFUND", recommended, order_id))
    elif refunded_total > 0 and refundable_total < 0.01:
        verdict = "refunded"
    else:
        expected = case.get("expected_total_brl")
        if expected is None:
            expected = case.get("order_total_brl")
        expected_value = as_float(expected)
        if expected_value is not None and abs(captured_total - expected_value) > 1.0:
            verdict = "capture_mismatch"
            diff = abs(captured_total - expected_value)
            causes.append(CandidateCause("PAYMENT_AMOUNT_MISMATCH", "platform", rank=1))
            if captured_total > expected_value:
                recommended = diff
                lines.append(RefundLine("OVERCHARGE_REFUND", recommended, order_id))
        else:
            verdict = "reconciled"

    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=ACTOR,
        target=COORDINATOR,
        decision_code=f"PAYMENT_{verdict.upper()}",
        attributes={
            "verdict": verdict,
            "captured_total_brl": round(captured_total, 2),
            "refunded_total_brl": round(refunded_total, 2),
            "recommended_refund_brl": round(recommended, 2),
        },
    )
    return PaymentResult(
        verdict=verdict,
        captured_total_brl=round(captured_total, 2),
        refunded_total_brl=round(refunded_total, 2),
        refundable_total_brl=round(refundable_total, 2),
        payment_references=_references(rows),
        financial_resolution=FinancialResolution(
            recommended_refund_brl=recommended, refund_lines=lines
        ),
        candidate_causes=causes,
        evidence_refs=list(dict.fromkeys(evidence_refs)),
    )
