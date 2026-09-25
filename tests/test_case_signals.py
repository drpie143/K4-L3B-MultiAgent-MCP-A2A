"""Evidence signals and the policy-driven verifier, on payloads shaped like live MCP data."""

from __future__ import annotations

from pathlib import Path

from student_agent.agents.case_signals import analyze, choose_primary
from student_agent.agents.entity_agent import EntityAgentResult
from student_agent.agents.order_shipment_agent import OrderShipmentResult
from student_agent.agents.payment_agent import PaymentAgentResult
from student_agent.agents.verifier_agent import verify_and_finalize
from student_agent.contracts import Contracts

ROOT = Path(__file__).resolve().parents[1]
ORDER = "64a8b3b4c19751b51d211deb32fdc67b"
SELLER = "seller-64a8b3b4c197"
REF = "ev_" + "a" * 32
CASE = {
    "case_id": "L3B_CASE_010",
    "opened_at": "2018-01-10T09:00:00-03:00",
    "customer_request": {
        "claimed_order_id": ORDER,
        "claims": [
            {"claim_id": "claim-010-a", "topic": "late_delivery_seller"},
            {"claim_id": "claim-010-b", "topic": "requested_full_refund"},
        ],
    },
    "candidate_order_ids": [ORDER, "candidate-010"],
}
# The order row is dated months after the complaint; the history also holds the version
# that matches the complaint window (seller handed over after the shipping limit).
ORDER_ROW = {
    "order_id": ORDER,
    "order_status": "canceled",
    "order_purchase_timestamp": "2018-05-02T09:00:00-03:00",
    "order_delivered_carrier_date": "2018-05-04T09:00:00-03:00",
    "order_delivered_customer_date": None,
    "order_estimated_delivery_date": "2018-05-12T09:00:00-03:00",
}
HISTORY = {
    "customer_unique_id": "customer-1deb32fdc67b",
    "orders": [
        ORDER_ROW,
        {
            "order_id": ORDER,
            "order_status": "delivered",
            "order_purchase_timestamp": "2017-12-29T09:00:00-03:00",
            "order_delivered_carrier_date": "2018-01-05T09:00:00-03:00",
            "order_delivered_customer_date": "2018-01-12T09:00:00-03:00",
            "order_estimated_delivery_date": "2018-01-08T09:00:00-03:00",
        },
    ],
}
ITEMS = [
    {"order_id": ORDER, "seller_id": SELLER, "shipping_limit_date": "2018-05-05T09:00:00-03:00"},
    {"order_id": ORDER, "seller_id": SELLER, "shipping_limit_date": "2018-01-01T09:00:00-03:00"},
]
SHIPMENT = {"order_id": ORDER, "events": []}
PAYMENTS = {
    "payments": [
        {"payment_sequential": "1", "payment_type": "credit_card", "payment_value": "79.00"},
        {"payment_sequential": "1", "payment_type": "credit_card", "payment_value": "18.00"},
    ],
    "events": [
        {"event_at": "2018-05-02T10:00:00-03:00", "event_type": "captured", "amount_brl": "79.00"},
        {"event_at": "2017-12-29T10:00:00-03:00", "event_type": "captured", "amount_brl": "18.00"},
    ],
}
POLICY = {
    "policy_version": "EC_POLICY_V2",
    "rules": {
        "late_delivery_seller": {
            "case_status": "action_required",
            "recommended_action": "refund_freight",
            "refund_brl": 18.0,
            "responsible_parties": [{"party_id": "seller-template", "party_type": "seller"}],
        },
        "canceled_order_paid": {
            "case_status": "action_required",
            "recommended_action": "issue_refund",
            "refund_brl": 79.0,
            "responsible_parties": [{"party_id": None, "party_type": "platform"}],
        },
    },
}


def test_rows_after_the_complaint_do_not_count_and_versions_use_their_own_limit() -> None:
    signals = analyze(CASE, ORDER, order=ORDER_ROW, history=HISTORY, items=ITEMS, payments=PAYMENTS)
    # The canceled row is dated after the complaint, so it is not a canceled-paid case.
    assert "canceled_order_paid" not in signals.detected
    assert "late_delivery_seller" in signals.detected
    assert signals.late_seller_ids == [SELLER]
    assert signals.shipment_verdict == "seller_delay"
    assert signals.captured_total == 18.0
    assert signals.status_conflict == ("customer", "order")


def test_identical_payment_rows_are_a_duplicate_but_a_split_is_not() -> None:
    split = {
        "payments": [
            {"payment_sequential": "1", "payment_type": "credit_card", "payment_value": "44.50"},
            {"payment_sequential": "2", "payment_type": "voucher", "payment_value": "44.50"},
        ]
    }
    assert "duplicate_charge" not in analyze(CASE, ORDER, payments=split).detected
    doubled = {"payments": split["payments"] * 2}
    signals = analyze(CASE, ORDER, payments=doubled)
    assert "duplicate_charge" in signals.detected
    assert signals.duplicate_amount == 44.5


def test_failed_refund_event_is_detected() -> None:
    refunds = {
        "events": [
            {
                "event_at": "2018-01-05T09:00:00-03:00",
                "event_type": "refund_requested",
                "amount_brl": "52.00",
                "status": "failed",
            }
        ]
    }
    assert "refund_failed" in analyze(CASE, ORDER, refunds=refunds).detected


def test_claim_confirmed_by_evidence_wins_and_unconfirmed_claim_yields_to_strong_signal() -> None:
    claims = ["late_delivery_seller", "requested_full_refund"]
    assert choose_primary(claims, {"late_delivery_seller", "valid_split_payment"}) == (
        "late_delivery_seller",
        0.92,
    )
    primary, confidence = choose_primary(claims, {"refund_failed"})
    assert primary == "refund_failed" and confidence < 0.7
    assert choose_primary(["unsupported_claim"], {"valid_split_payment"})[0] == "unsupported_claim"


def test_verifier_takes_refund_action_and_parties_from_policy() -> None:
    entity = EntityAgentResult(
        status="resolved",
        resolved_order_ids=[ORDER],
        rejected_candidates=["candidate-010"],
        confidence=0.8,
        customer_unique_id="customer-1deb32fdc67b",
        evidence_refs=[REF],
        order_evidence={ORDER: {"evidence_ref": REF, "data": ORDER_ROW}},
        history_evidence={"evidence_ref": REF, "data": HISTORY},
    )
    shipment = OrderShipmentResult(
        verdict="insufficient_evidence",
        seller_ids=[SELLER],
        raw={"get_order_items": ITEMS, "get_shipment_summary": SHIPMENT},
    )
    payment = PaymentAgentResult(verdict="duplicate_capture", raw={"payments": PAYMENTS})
    output = verify_and_finalize(CASE, entity, shipment, payment, POLICY)

    Contracts(ROOT / "contracts" / "schemas").validate_output(output, CASE["case_id"])
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["assessment"]["case_status"] == "action_required"
    assert output["financial_resolution"]["recommended_refund_brl"] == 18.0
    assert output["resolution_actions"] == ["refund_freight"]
    assert output["shipment_analysis"]["late_seller_ids"] == [SELLER]
    assert output["payment_analysis"]["verdict"] == "reconciled"
    # The policy's template seller id is replaced by the seller proven late.
    assert output["root_cause_analysis"]["responsible_parties"] == [
        {"party_type": "seller", "party_id": SELLER}
    ]


def test_reconciliation_mismatch_event_confirms_a_mismatch_claim() -> None:
    payments = {
        "payments": [
            {"payment_sequential": "1", "payment_type": "credit_card", "payment_value": "35.00"}
        ],
        "events": [
            {"event_at": "2018-01-02T10:00:00-03:00", "event_type": "captured", "amount_brl": "35"},
            {
                "event_at": "2018-01-02T12:00:00-03:00",
                "event_type": "reconciliation_mismatch",
                "amount_brl": "35.00",
                "status": "open",
            },
        ],
    }
    detected = analyze(CASE, ORDER, payments=payments).detected
    assert "payment_mismatch" in detected
    assert choose_primary(["payment_mismatch", "requested_full_refund"], detected)[0] == (
        "payment_mismatch"
    )


def test_policy_is_fetched_once_per_session_and_never_cited(tmp_path: Path) -> None:
    import asyncio
    import hashlib

    from student_agent.trace import TraceWriter
    from student_agent.workflow import solve_case

    def envelope(domain: str, data: object, tag: str) -> dict:
        digest = hashlib.sha256(tag.encode()).hexdigest()
        return {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": f"ev_{digest[:32]}",
            "result_hash": f"sha256:{digest}",
            "domain": domain,
            "data": data,
        }

    policy = envelope("policy", POLICY, "policy")

    class Gateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def list_tools(self) -> list[str]:
            return ["get_order", "get_policy"]

        async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict:
            self.calls.append((tool_name, case_id))
            if tool_name == "get_policy":
                return policy
            raise RuntimeError("not found")

    gateway = Gateway()
    trace = TraceWriter(tmp_path / "trace.jsonl", Contracts(ROOT / "contracts" / "schemas"))

    async def run_two() -> list[dict]:
        cases = [
            dict(CASE, case_id=f"L3B_CASE_00{i}", policy_version="EC_POLICY_V2") for i in (1, 2)
        ]
        return list(await asyncio.gather(*(solve_case(c, gateway, trace) for c in cases)))

    outputs = asyncio.run(run_two())
    assert [c for c in gateway.calls if c[0] == "get_policy"] == [("get_policy", "L3B_CASE_001")]
    assert all(policy["evidence_ref"] not in o["evidence_refs"] for o in outputs)
    assert policy["evidence_ref"] not in trace.path.read_text(encoding="utf-8")
