"""Conflict resolution and final verification.

Specialists contribute verdicts and candidate causes. This module chooses one
primary issue, records source conflicts explicitly, and emits a schema-shaped
case output.
"""

from __future__ import annotations

from typing import Any

from .. import OUTPUT_SCHEMA_VERSION
from ..models.messages import EntityResult, PaymentResult, ShipmentResult
from ..utils.evidence import as_float, first_text, round_money, unique_ids, valid_evidence_refs
from .case_signals import MONEY_EPS, analyze, choose_primary

PRIORITY = (
    "duplicate_charge",
    "payment_mismatch",
    "refund_failed",
    "refund_pending",
    "canceled_order_paid",
    "unavailable_order_paid",
    "late_delivery_seller",
    "late_delivery_logistics",
    "valid_split_payment",
)

SHIPMENT_CAUSES = {
    "lost": ("SHIPMENT_LOST", "logistics_provider"),
    "returned": ("SHIPMENT_RETURNED", "logistics_provider"),
    "seller_delay": ("SELLER_SHIPMENT_DELAY", "seller"),
    "logistics_delay": ("LOGISTICS_DELIVERY_DELAY", "logistics_provider"),
    "conflicting": ("SHIPMENT_SOURCE_CONFLICT", "unknown"),
}
PAYMENT_CAUSES = {
    "duplicate_capture": ("DUPLICATE_CHARGE", "payment_provider"),
    "capture_mismatch": ("PAYMENT_AMOUNT_MISMATCH", "platform"),
    "refund_failed": ("REFUND_GATEWAY_FAILURE", "payment_provider"),
    "refund_pending": ("REFUND_PENDING_SETTLEMENT", "payment_provider"),
}
PRIMARY_CAUSES = {
    "canceled_order_paid": ("CANCELED_ORDER_CAPTURED", "platform"),
    "unavailable_order_paid": ("UNAVAILABLE_ORDER_CAPTURED", "platform"),
    "valid_split_payment": ("VALID_SPLIT_PAYMENT", "payment_provider"),
    "unsupported_claim": ("UNSUPPORTED_CUSTOMER_CLAIM", "customer"),
    "insufficient_evidence": ("INSUFFICIENT_EVIDENCE", "unknown"),
    "duplicate_charge": ("DUPLICATE_CHARGE", "payment_provider"),
    "payment_mismatch": ("PAYMENT_AMOUNT_MISMATCH", "platform"),
    "refund_failed": ("REFUND_GATEWAY_FAILURE", "payment_provider"),
    "refund_pending": ("REFUND_PENDING_SETTLEMENT", "payment_provider"),
    "late_delivery_seller": ("SELLER_SHIPMENT_DELAY", "seller"),
    "late_delivery_logistics": ("LOGISTICS_DELIVERY_DELAY", "logistics_provider"),
}
REFUND_REASONS = {
    "duplicate_charge": "DUPLICATE_CHARGE_REFUND",
    "payment_mismatch": "OVERCHARGE_REFUND",
    "refund_failed": "REFUND_RETRY",
    "refund_pending": "REFUND_PENDING",
    "canceled_order_paid": "CANCELED_ORDER_REFUND",
    "unavailable_order_paid": "UNAVAILABLE_ORDER_REFUND",
    "late_delivery_logistics": "UNDELIVERED_ORDER_REFUND",
}
OPERATIONAL_ACTIONS = {
    "duplicate_charge": "refund_duplicate_capture",
    "payment_mismatch": "refund_capture_difference",
    "refund_failed": "retry_failed_refund",
    "refund_pending": "complete_pending_refund",
    "canceled_order_paid": "refund_canceled_order",
    "unavailable_order_paid": "refund_unavailable_order",
    "late_delivery_seller": "record_seller_delay",
    "late_delivery_logistics": "record_logistics_delay",
}
PARTIES = {"seller", "platform", "logistics_provider", "payment_provider", "customer", "unknown"}
MONEY = 1.0


def _clamp_unit(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(min(1.0, max(0.0, number)), 2)


def _kind(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"lost", "missing"} or text.endswith("_lost"):
        return "lost"
    if "return" in text:
        return "returned"
    if text in {"delivered"} or text.endswith("_delivered"):
        return "delivered"
    return text


def _order_status(entity: EntityResult, shipment: ShipmentResult) -> str | None:
    reported = getattr(shipment, "order_status", None)
    if isinstance(reported, str) and reported.strip():
        return reported.strip().lower()
    evidence = getattr(entity, "order_evidence", {})
    if not isinstance(evidence, dict):
        return None
    for payload in evidence.values():
        data = payload.get("data") if isinstance(payload, dict) else payload
        text = first_text(data, ("order_status",))
        if text:
            return text.lower()
    return None


def _shipment_view(shipment: ShipmentResult) -> tuple[str, list[str], bool]:
    verdict = shipment.verdict
    late = unique_ids(list(shipment.late_seller_ids))
    if verdict == "seller_delay" and not late:
        late = unique_ids(list(shipment.seller_ids))
    if verdict == "seller_delay" and not late:
        verdict = "insufficient_evidence"
    if verdict != "seller_delay":
        late = []
    timeline = bool(shipment.timeline_complete) and verdict != "insufficient_evidence"
    return verdict, late, timeline


def _payment_view(payment: PaymentResult) -> str:
    if payment.verdict == "refunded" and not (payment.refunded_total_brl or 0) > 0:
        if payment.captured_total_brl is None:
            return "insufficient_evidence"
        return "reconciled"
    return payment.verdict


def _remainder(payment: PaymentResult) -> float:
    if payment.refundable_total_brl is not None:
        return max(0.0, float(payment.refundable_total_brl))
    if payment.captured_total_brl is None:
        return 0.0
    refunded = payment.refunded_total_brl or 0.0
    return max(0.0, float(payment.captured_total_brl) - float(refunded))


def _clean_conflict(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    field = str(raw.get("field") or "").strip()
    sources = unique_ids(list(raw.get("sources") or []), limit=5)
    if not field or len(field) > 100 or len(sources) < 2:
        return None
    selected = raw.get("selected_source")
    if selected is not None:
        selected = str(selected)
        if selected not in sources:
            selected = None
    code = str(raw.get("resolution_code") or "UNRESOLVED_CONFLICT").strip() or "UNRESOLVED_CONFLICT"
    return {
        "field": field,
        "sources": sources,
        "selected_source": selected,
        "resolution_code": code[:80],
    }


def _conflicts(
    entity: EntityResult,
    shipment: ShipmentResult,
    payment: PaymentResult,
    order_status: str | None,
    ship_verdict: str,
) -> list[dict[str, Any]]:
    raw_conflicts = list(getattr(shipment, "conflicts", []) or [])
    shipment_status = getattr(shipment, "shipment_status", None)
    order_kind = _kind(order_status)
    ship_kind = _kind(shipment_status if isinstance(shipment_status, str) else None)
    if order_kind == "delivered" and ship_kind in {"lost", "returned"}:
        raw_conflicts.append(
            {
                "field": "shipment_status",
                "sources": ["order", "shipment"],
                "selected_source": "shipment",
                "resolution_code": "SHIPMENT_SOURCE_PRECEDENCE",
            }
        )
    order_total = getattr(shipment, "order_total_brl", None)
    captured = payment.captured_total_brl
    if (
        isinstance(order_total, int | float)
        and captured is not None
        and abs(float(captured) - float(order_total)) > MONEY
    ):
        raw_conflicts.append(
            {
                "field": "captured_total_brl",
                "sources": ["order_items", "payment"],
                "selected_source": "payment",
                "resolution_code": "PAYMENT_CAPTURE_PRECEDENCE",
            }
        )
    if ship_verdict == "conflicting" and not raw_conflicts:
        raw_conflicts.append(
            {
                "field": "shipment_status",
                "sources": ["order", "shipment"],
                "selected_source": None,
                "resolution_code": "UNRESOLVED_CONFLICT",
            }
        )
    cleaned: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_conflicts:
        conflict = _clean_conflict(raw)
        if conflict is None or conflict["field"] in seen:
            continue
        seen.add(conflict["field"])
        cleaned.append(conflict)
        if len(cleaned) == 5:
            break
    del entity
    return cleaned


def _detected(
    payment: PaymentResult,
    pay_verdict: str,
    ship_verdict: str,
    order_status: str | None,
) -> set[str]:
    found: set[str] = set()
    if pay_verdict == "duplicate_capture":
        found.add("duplicate_charge")
    if pay_verdict == "refund_failed":
        found.add("refund_failed")
    if pay_verdict == "refund_pending":
        found.add("refund_pending")
    captured = payment.captured_total_brl
    if pay_verdict == "capture_mismatch":
        found.add("payment_mismatch")
    if order_status == "canceled" and captured is not None and captured > 0:
        found.add("canceled_order_paid")
    if order_status == "unavailable" and captured is not None and captured > 0:
        found.add("unavailable_order_paid")
    if ship_verdict == "seller_delay":
        found.add("late_delivery_seller")
    if ship_verdict in {"logistics_delay", "lost", "returned"}:
        found.add("late_delivery_logistics")
    return found


def _split_payment(payment: PaymentResult, shipment: ShipmentResult, pay_verdict: str) -> bool:
    total = getattr(shipment, "order_total_brl", None)
    captured = payment.captured_total_brl
    if pay_verdict != "reconciled" or not isinstance(total, int | float) or captured is None:
        return False
    return len(payment.payment_references) >= 2 and abs(float(captured) - float(total)) <= MONEY


def _totals_mismatch(payment: PaymentResult, shipment: ShipmentResult, pay_verdict: str) -> bool:
    total = getattr(shipment, "order_total_brl", None)
    captured = payment.captured_total_brl
    if pay_verdict not in {"reconciled", "insufficient_evidence"}:
        return False
    if not isinstance(total, int | float) or captured is None:
        return False
    return abs(float(captured) - float(total)) > MONEY


def _recommend(
    primary: str,
    payment: PaymentResult,
    shipment: ShipmentResult,
    ship_verdict: str,
) -> float:
    specialist = float(payment.financial_resolution.recommended_refund_brl or 0)
    remainder = _remainder(payment)
    item_total = getattr(shipment, "order_total_brl", None)
    captured = payment.captured_total_brl
    if primary == "duplicate_charge":
        recommended = specialist if specialist > 0 else remainder
    elif primary == "payment_mismatch":
        if specialist > 0 and payment.verdict == "capture_mismatch":
            recommended = specialist
        elif (
            isinstance(item_total, int | float)
            and captured is not None
            and captured > float(item_total)
        ):
            recommended = float(captured) - float(item_total)
        else:
            recommended = 0.0
    elif primary in {"refund_failed", "refund_pending"}:
        if specialist > 0:
            recommended = specialist
        elif primary == "refund_failed":
            recommended = remainder
        else:
            recommended = 0.0
    elif primary in {"canceled_order_paid", "unavailable_order_paid"} or ship_verdict in {
        "lost",
        "returned",
    }:
        recommended = remainder
    else:
        recommended = 0.0
    if payment.refundable_total_brl is not None:
        recommended = min(recommended, float(payment.refundable_total_brl))
    return round(max(0.0, recommended), 2)


def _confidence(
    entity: EntityResult, primary: str, ship_verdict: str, timeline_complete: bool
) -> float:
    unresolved = entity.status != "resolved" or primary == "insufficient_evidence"
    if unresolved or ship_verdict == "conflicting":
        return _clamp_unit(min(entity.confidence, 0.35), 0.35)
    if not timeline_complete and primary in {"late_delivery_seller", "late_delivery_logistics"}:
        return _clamp_unit(min(0.6, max(0.45, entity.confidence * 0.7)), 0.5)
    return _clamp_unit(min(0.92, max(0.7, entity.confidence)), 0.7)


def _causes(
    primary: str, ship_verdict: str, pay_verdict: str, late_sellers: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seller_id = late_sellers[0] if late_sellers else None
    ordered: list[tuple[str, str, str | None]] = []

    def add(code: str, party: str, party_id: str | None = None) -> None:
        if party not in PARTIES:
            party = "unknown"
        if any(item[0] == code for item in ordered):
            return
        ordered.append((code, party, party_id if party_id else None))

    shipment_cause = SHIPMENT_CAUSES.get(ship_verdict)
    if shipment_cause is not None:
        party_id = seller_id if shipment_cause[1] == "seller" else None
        add(shipment_cause[0], shipment_cause[1], party_id)
    payment_cause = PAYMENT_CAUSES.get(pay_verdict)
    if payment_cause is not None:
        add(*payment_cause)
    primary_cause = PRIMARY_CAUSES.get(primary)
    if primary_cause is not None:
        party_id = seller_id if primary_cause[1] == "seller" else None
        add(primary_cause[0], primary_cause[1], party_id)
    if not ordered:
        add("INSUFFICIENT_EVIDENCE", "unknown")
    ranked = [
        {"cause_code": code, "rank": index}
        for index, (code, _, _) in enumerate(ordered[:5], start=1)
    ]
    parties = []
    seen: set[tuple[str, str | None]] = set()
    for _, party, party_id in ordered[:5]:
        key = (party, party_id)
        if key in seen:
            continue
        seen.add(key)
        parties.append({"party_type": party, "party_id": party_id})
    return ranked, parties


def _claims(
    case: dict[str, Any],
    primary: str,
    recommended: float,
    refundable: float | None,
    confidence: float,
    evidence_refs: list[str],
) -> list[dict[str, Any]]:
    request = case.get("customer_request")
    claims = request.get("claims") if isinstance(request, dict) else None
    if not isinstance(claims, list):
        return []
    assessments: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_id = claim.get("claim_id")
        topic = claim.get("topic")
        if not isinstance(claim_id, str) or not claim_id or not isinstance(topic, str):
            continue
        if primary == "insufficient_evidence":
            verdict = "insufficient_evidence"
            claim_confidence = 0.35
        elif topic == "requested_full_refund":
            if recommended <= 0:
                verdict = "unsupported"
            elif refundable is None or recommended + 0.01 >= refundable * 0.9:
                verdict = "supported"
            else:
                verdict = "partially_supported"
            claim_confidence = confidence if verdict == "supported" else min(confidence, 0.8)
        elif topic == primary:
            verdict = "supported"
            claim_confidence = confidence
        else:
            verdict = "unsupported"
            claim_confidence = min(confidence, 0.8)
        assessments.append(
            {
                "claim_id": claim_id[:64],
                "verdict": verdict,
                "confidence": _clamp_unit(claim_confidence, 0.5),
                "evidence_refs": evidence_refs[:30],
            }
        )
        if len(assessments) == 5:
            break
    return assessments


def verify_and_finalize(
    case: dict[str, Any],
    entity: EntityResult,
    shipment: ShipmentResult,
    payment: PaymentResult,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check cross-field invariants and return the L3B case output.

    With the case's policy (``get_policy`` data) and a resolved order, the decision comes
    from evidence signals checked against the complaint timeline, and the refund, status,
    action and responsible parties come from the matching policy rule.
    """
    output = _rule_based_output(case, entity, shipment, payment)
    if policy is not None:
        output = _apply_policy(output, case, entity, shipment, payment, policy)
    return output


PAYMENT_VERDICT_BY_ISSUE = {
    "duplicate_charge": "duplicate_capture",
    "payment_mismatch": "capture_mismatch",
    "refund_failed": "refund_failed",
    "refund_pending": "refund_pending",
}
DELIVERY_ISSUES = {"late_delivery_seller", "late_delivery_logistics"}


def _apply_policy(
    output: dict[str, Any],
    case: dict[str, Any],
    entity: EntityResult,
    shipment: ShipmentResult,
    payment: PaymentResult,
    policy: dict[str, Any],
) -> dict[str, Any]:
    rules = policy.get("rules") if isinstance(policy, dict) else None
    resolved = output["entity_resolution"]["resolved_order_ids"]
    if not isinstance(rules, dict) or output["entity_resolution"]["status"] != "resolved":
        return output
    order_id = resolved[0]
    order_payload = (getattr(entity, "order_evidence", {}) or {}).get(order_id) or {}
    history_payload = getattr(entity, "history_evidence", None) or {}
    ship_raw = getattr(shipment, "raw", {}) or {}
    pay_raw = getattr(payment, "raw", {}) or {}
    signals = analyze(
        case,
        order_id,
        order=order_payload.get("data"),
        history=history_payload.get("data"),
        items=ship_raw.get("get_order_items"),
        shipment=ship_raw.get("get_shipment_summary"),
        payments=pay_raw.get("payments"),
        refunds=pay_raw.get("refunds"),
    )
    request = case.get("customer_request")
    claims = request.get("claims") if isinstance(request, dict) else None
    topics = [str(c.get("topic")) for c in claims or [] if isinstance(c, dict) and c.get("topic")]
    primary, confidence = choose_primary(topics, signals.detected)
    rule = rules.get(primary)
    if not isinstance(rule, dict):
        return output

    case_status = str(rule.get("case_status") or output["assessment"]["case_status"])
    if case_status not in {"action_required", "no_action", "needs_investigation"}:
        case_status = output["assessment"]["case_status"]
    refund = round_money(as_float(rule.get("refund_brl"))) or 0.0
    action = rule.get("recommended_action")
    actions = [str(action)[:80]] if isinstance(action, str) and action else []
    if case_status == "action_required" and not actions:
        actions = [OPERATIONAL_ACTIONS.get(primary, "review_case")]
    if case_status == "no_action":
        refund = 0.0

    # Shipment: follow evidence; a delivery primary issue must name its verdict.
    ship_verdict = signals.shipment_verdict
    late = list(signals.late_seller_ids)
    if primary == "late_delivery_seller":
        ship_verdict = "seller_delay"
        late = late or signals.seller_ids[:1]
    elif primary == "late_delivery_logistics" and ship_verdict not in {"lost", "returned"}:
        ship_verdict = "logistics_delay"
    if ship_verdict != "seller_delay":
        late = []
    if ship_verdict == "seller_delay" and not late:
        ship_verdict = "insufficient_evidence"

    captured = signals.captured_total
    refunded = signals.refunded_total
    refundable = None if captured is None else round(max(0.0, captured - (refunded or 0.0)), 2)
    pay_verdict = PAYMENT_VERDICT_BY_ISSUE.get(primary)
    if pay_verdict is None:
        if captured is None:
            pay_verdict = "insufficient_evidence"
        elif (refunded or 0) > 0 and (refundable or 0) < MONEY_EPS:
            pay_verdict = "refunded"
        else:
            pay_verdict = "reconciled"

    parties: list[dict[str, Any]] = []
    for party in rule.get("responsible_parties") or []:
        if not isinstance(party, dict):
            continue
        party_type = party.get("party_type") if party.get("party_type") in PARTIES else "unknown"
        party_id = party.get("party_id")
        if party_type == "seller":
            party_id = (late or signals.seller_ids or [party_id])[0]
        entry = {"party_type": party_type, "party_id": party_id}
        if entry not in parties:
            parties.append(entry)
    cause = PRIMARY_CAUSES.get(primary, ("INSUFFICIENT_EVIDENCE", "unknown"))
    if not parties:
        parties = [{"party_type": cause[1], "party_id": late[0] if late else None}]

    # Every case carries one decoy timeline built from another issue type; signals outside
    # the confirmed claim are that decoy, not a second issue.
    secondary: list[str] = []
    conflicts: list[dict[str, Any]] = []
    if signals.status_conflict is not None:
        selected, other = signals.status_conflict
        conflicts.append(
            {
                "field": "order_status",
                "sources": [other, selected],
                "selected_source": selected,
                "resolution_code": "COMPLAINT_TIMELINE_PRECEDENCE",
            }
        )

    output["assessment"] = {
        "primary_issue": primary,
        "secondary_issues": secondary[:10],
        "case_status": case_status,
        "confidence": _clamp_unit(confidence if entity.confidence >= 0.7 else 0.5, 0.5),
    }
    output["affected_entities"]["seller_ids"] = unique_ids(
        [*signals.seller_ids, *late, *output["affected_entities"]["seller_ids"]]
    )
    output["shipment_analysis"] = {
        "verdict": ship_verdict,
        "late_seller_ids": late,
        "timeline_complete": bool(
            signals.timeline_complete and ship_verdict != "insufficient_evidence"
        ),
    }
    output["payment_analysis"] = {
        "verdict": pay_verdict,
        "captured_total_brl": captured,
        "refunded_total_brl": refunded,
        "refundable_total_brl": refundable,
    }
    if refundable is not None:
        refund = min(refund, refundable) if refundable > 0 else refund
    output["financial_resolution"] = {
        "currency": "BRL",
        "recommended_refund_brl": refund,
        "refund_lines": (
            [{"reason_code": primary.upper(), "amount_brl": refund, "entity_id": order_id}]
            if refund > 0
            else []
        ),
    }
    output["root_cause_analysis"] = {
        "ranked_causes": [{"cause_code": cause[0], "rank": 1}],
        "responsible_parties": parties[:5],
    }
    output["data_conflicts"] = conflicts
    output["resolution_actions"] = actions[:8]
    claims_out = _claims(
        case,
        primary,
        refund,
        refundable,
        output["assessment"]["confidence"],
        output["evidence_refs"],
    )
    if claims_out:
        output["claim_assessments"] = claims_out
    return output


def _rule_based_output(
    case: dict[str, Any],
    entity: EntityResult,
    shipment: ShipmentResult,
    payment: PaymentResult,
) -> dict[str, Any]:
    resolved = unique_ids(list(entity.resolved_order_ids))
    rejected = [
        item for item in unique_ids(list(entity.rejected_candidates)) if item not in set(resolved)
    ]
    entity_resolved = entity.status == "resolved" and bool(resolved)
    ship_verdict, late_sellers, timeline_complete = _shipment_view(shipment)
    pay_verdict = _payment_view(payment)
    order_status = _order_status(entity, shipment)
    if not entity_resolved:
        ship_verdict = "insufficient_evidence"
        pay_verdict = "insufficient_evidence"
        late_sellers = []
        timeline_complete = False

    found: set[str] = set()
    if entity_resolved:
        found = _detected(payment, pay_verdict, ship_verdict, order_status)
    if entity_resolved and _totals_mismatch(payment, shipment, pay_verdict):
        found.add("payment_mismatch")
    if entity_resolved and _split_payment(payment, shipment, pay_verdict):
        found.add("valid_split_payment")

    primary = next((issue for issue in PRIORITY if issue in found), None)
    delivery_known = ship_verdict not in {"insufficient_evidence", "conflicting"}
    payment_known = pay_verdict != "insufficient_evidence"
    if primary is None:
        primary = (
            "unsupported_claim"
            if entity_resolved and delivery_known and payment_known
            else "insufficient_evidence"
        )
    if ship_verdict == "conflicting" and primary == "unsupported_claim":
        primary = "insufficient_evidence"

    recommended = _recommend(primary, payment, shipment, ship_verdict) if entity_resolved else 0.0
    if primary in {"late_delivery_seller", "late_delivery_logistics"} and ship_verdict not in {
        "lost",
        "returned",
    }:
        recommended = 0.0
    if pay_verdict == "insufficient_evidence" and primary in {
        "duplicate_charge",
        "payment_mismatch",
        "refund_failed",
        "refund_pending",
    }:
        primary = "insufficient_evidence"
        recommended = 0.0

    if primary == "insufficient_evidence":
        case_status = "needs_investigation"
        recommended = 0.0
    elif recommended > 0 or primary in {
        "refund_failed",
        "refund_pending",
        "late_delivery_seller",
        "late_delivery_logistics",
    }:
        case_status = "action_required"
    else:
        case_status = "no_action"
        recommended = 0.0

    actions: list[str] = []
    if case_status == "action_required":
        if ship_verdict == "lost":
            actions.append("investigate_lost_shipment")
        elif ship_verdict == "returned":
            actions.append("process_return")
        action = OPERATIONAL_ACTIONS.get(primary)
        if action:
            actions.append(action)
        if ship_verdict in {"lost", "returned"} and recommended > 0:
            actions.append("refund_undelivered_order")
        if not actions:
            actions.append("review_case")
    recommended = round(recommended, 2)
    if payment.refundable_total_brl is not None and primary != "insufficient_evidence":
        recommended = min(recommended, round(max(0.0, float(payment.refundable_total_brl)), 2))
    if case_status == "no_action":
        recommended = 0.0
        actions = []

    order_id = resolved[0] if resolved else None
    refund_lines: list[dict[str, Any]] = []
    if recommended > 0:
        refund_lines.append(
            {
                "reason_code": REFUND_REASONS.get(primary, "REVIEWED_REFUND"),
                "amount_brl": recommended,
                "entity_id": order_id,
            }
        )

    if primary == "duplicate_charge":
        pay_verdict = "duplicate_capture"
    elif primary == "payment_mismatch":
        pay_verdict = "capture_mismatch"
    elif primary == "refund_failed":
        pay_verdict = "refund_failed"
    elif primary == "refund_pending":
        pay_verdict = "refund_pending"

    secondary = [issue for issue in PRIORITY if issue in found and issue != primary]
    if ship_verdict == "lost":
        secondary.append("shipment_lost")
    elif ship_verdict == "returned":
        secondary.append("shipment_returned")
    secondary = unique_ids(secondary, limit=10)

    evidence_refs = valid_evidence_refs(
        [
            *entity.evidence_refs,
            *shipment.evidence_refs,
            *payment.evidence_refs,
        ]
    )
    conflicts = _conflicts(entity, shipment, payment, order_status, ship_verdict)
    confidence = _confidence(entity, primary, ship_verdict, timeline_complete)
    ranked, parties = _causes(primary, ship_verdict, pay_verdict, late_sellers)
    claims = _claims(
        case,
        primary,
        recommended,
        None if payment.refundable_total_brl is None else float(payment.refundable_total_brl),
        confidence,
        evidence_refs,
    )

    captured = round_money(payment.captured_total_brl)
    refunded = round_money(payment.refunded_total_brl)
    refundable = round_money(payment.refundable_total_brl)
    if not entity_resolved:
        captured = refunded = refundable = None
        pay_verdict = "insufficient_evidence"
        ship_verdict = "insufficient_evidence"

    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "case_id": case.get("case_id"),
        "assessment": {
            "primary_issue": primary,
            "secondary_issues": secondary,
            "case_status": case_status,
            "confidence": confidence,
        },
        "affected_entities": {
            "order_ids": resolved,
            "item_ids": unique_ids(list(shipment.item_ids)),
            "seller_ids": unique_ids([*shipment.seller_ids, *late_sellers]),
            "payment_references": unique_ids(list(payment.payment_references)),
            "shipment_ids": unique_ids(list(shipment.shipment_ids)),
        },
        "entity_resolution": {
            "status": (
                entity.status
                if entity.status in {"resolved", "ambiguous", "not_found"}
                else "not_found"
            ),
            "resolved_order_ids": resolved,
            "rejected_candidates": rejected,
            "confidence": _clamp_unit(entity.confidence, 0.0),
        },
        "customer_context": {
            "customer_unique_id": (
                entity.customer_unique_id if isinstance(entity.customer_unique_id, str) else None
            ),
            "related_order_ids": [
                item
                for item in unique_ids(list(entity.related_order_ids))
                if item not in set(resolved)
            ],
        },
        "shipment_analysis": {
            "verdict": ship_verdict,
            "late_seller_ids": late_sellers,
            "timeline_complete": timeline_complete,
        },
        "payment_analysis": {
            "verdict": pay_verdict,
            "captured_total_brl": captured,
            "refunded_total_brl": refunded,
            "refundable_total_brl": refundable,
        },
        "root_cause_analysis": {"ranked_causes": ranked, "responsible_parties": parties},
        "evidence_refs": evidence_refs,
        "data_conflicts": conflicts,
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": recommended,
            "refund_lines": refund_lines,
        },
        "resolution_actions": list(dict.fromkeys(actions))[:8],
    }
    if claims:
        output["claim_assessments"] = claims
    return output
