"""Coordinator, conflict resolution and verifier."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.agents.verifier_agent import verify_and_finalize
from student_agent.contracts import Contracts
from student_agent.models.messages import (
    EntityResult,
    FinancialResolution,
    PaymentResult,
    RefundLine,
    ShipmentResult,
)
from student_agent.trace import TraceWriter
from student_agent.workflow import solve_case

ROOT = Path(__file__).resolve().parents[1]
ORDER = "a" * 32
FAKE = "candidate-x"
CUSTOMER = "f" * 32
SELLER = "seller-1"


def evidence(domain: str, data: Any, tag: str) -> dict[str, Any]:
    digest = hashlib.sha256(tag.encode()).hexdigest()
    return {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": f"ev_{digest[:32]}",
        "result_hash": f"sha256:{digest}",
        "domain": domain,
        "data": data,
    }


class Gateway:
    def __init__(
        self,
        responses: dict[tuple[str, str], dict[str, Any]],
        *,
        errors: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.responses = responses
        self.errors = errors or {}
        self.calls: list[tuple[str, str]] = []

    async def list_tools(self) -> list[str]:
        names = {tool for tool, _ in self.responses} | {tool for tool, _ in self.errors}
        return sorted(names)

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        key = next(iter(arguments.values()))
        self.calls.append((tool_name, key))
        if (tool_name, key) in self.errors:
            raise RuntimeError(self.errors[(tool_name, key)])
        try:
            return self.responses[(tool_name, key)]
        except KeyError as exc:
            raise RuntimeError(f"MCP tool {tool_name} failed: missing") from exc


@pytest.fixture
def trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", Contracts(ROOT / "contracts" / "schemas"))


def events(trace: TraceWriter) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]


def validate(output: dict[str, Any]) -> None:
    Contracts(ROOT / "contracts" / "schemas").validate_output(output, output["case_id"])


def order_row(status: str = "delivered") -> dict[str, Any]:
    return {
        "order_id": ORDER,
        "customer_unique_id": CUSTOMER,
        "order_status": status,
        "order_purchase_timestamp": "2018-01-01 10:00:00",
        "order_delivered_carrier_date": "2018-01-09 10:00:00",
        "order_delivered_customer_date": "2018-01-15 10:00:00",
        "order_estimated_delivery_date": "2018-01-20 10:00:00",
    }


def items(limit: str = "2018-01-10 10:00:00", price: float = 80.0, freight: float = 20.0) -> dict:
    return {
        "items": [
            {
                "order_item_id": "item-1",
                "product_id": "product-1",
                "seller_id": SELLER,
                "shipping_limit_date": limit,
                "price": price,
                "freight_value": freight,
            }
        ]
    }


def shipment(
    status: str = "delivered",
    carrier: str = "2018-01-09 10:00:00",
    customer_at: str = "2018-01-15 10:00:00",
    estimated: str = "2018-01-20 10:00:00",
) -> dict[str, Any]:
    return {
        "shipment_id": "shp-1",
        "shipment_status": status,
        "order_delivered_carrier_date": carrier,
        "order_delivered_customer_date": customer_at,
        "order_estimated_delivery_date": estimated,
    }


def payments(*rows: dict[str, Any]) -> list[dict[str, Any]]:
    return list(rows)


def case(case_id: str, topics: list[str]) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "opened_at": "2018-02-01T09:00:00-03:00",
        "customer_request": {
            "claimed_order_id": ORDER,
            "claims": [
                {"claim_id": f"{case_id}-{index}", "topic": topic}
                for index, topic in enumerate(topics, start=1)
            ],
        },
        "candidate_order_ids": [ORDER, FAKE],
        "policy_version": "EC_POLICY_V2",
        "investigation_scope": {
            "include_customer_history": True,
            "include_product_context": False,
        },
    }


def gateway_for(
    *,
    order_status: str = "delivered",
    shipment_payload: dict[str, Any] | None = None,
    payment_rows: list[dict[str, Any]] | None = None,
    item_payload: dict[str, Any] | None = None,
) -> Gateway:
    history = evidence(
        "customer",
        {"customer_unique_id": CUSTOMER, "orders": [{"order_id": ORDER}]},
        "history",
    )
    return Gateway(
        {
            ("get_order", ORDER): evidence(
                "order", order_row(order_status), f"order-{order_status}"
            ),
            ("get_customer_history", CUSTOMER): history,
            ("get_order_items", ORDER): evidence("item", item_payload or items(), "items"),
            ("get_shipment_summary", ORDER): evidence(
                "shipment", shipment_payload or shipment(), "shipment"
            ),
            ("get_order_payments", ORDER): evidence("payment", payment_rows or [], "payments"),
        },
        errors={("get_order", FAKE): "not in case scope"},
    )


def run(case_payload: dict[str, Any], gateway: Gateway, trace: TraceWriter) -> dict[str, Any]:
    return asyncio.run(solve_case(case_payload, gateway, trace))  # type: ignore[arg-type]


def test_unresolved_entity_does_not_call_specialists(trace: TraceWriter) -> None:
    gateway = Gateway(
        {},
        errors={("get_order", "missing-order"): "not in case scope"},
    )
    output = run(
        {
            "case_id": "L3B_CASE_099",
            "candidate_order_ids": ["missing-order"],
            "customer_request": {"claimed_order_id": "missing-order", "claims": []},
        },
        gateway,
        trace,
    )

    validate(output)
    assert output["assessment"]["primary_issue"] == "insufficient_evidence"
    assert output["assessment"]["case_status"] == "needs_investigation"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0
    assert output["entity_resolution"]["status"] == "not_found"
    assert "get_order_items" not in {tool for tool, _ in gateway.calls}
    assert "get_order_payments" not in {tool for tool, _ in gateway.calls}
    event_types = {event["event_type"] for event in events(trace)}
    assert {"task_assigned", "handoff", "verification_completed"} <= event_types


def test_seller_delay_does_not_turn_payment_into_a_refund(trace: TraceWriter) -> None:
    gateway = gateway_for(
        shipment_payload=shipment(carrier="2018-01-20 10:00:00", estimated="2018-01-28 10:00:00"),
        item_payload=items(limit="2018-01-10 10:00:00"),
        payment_rows=[
            {
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_value": 100.0,
                "payment_reference": "PAY-1",
            }
        ],
    )
    output = run(
        case("L3B_CASE_010", ["late_delivery_seller", "requested_full_refund"]),
        gateway,
        trace,
    )

    validate(output)
    assert output["assessment"]["primary_issue"] == "late_delivery_seller"
    assert output["assessment"]["case_status"] == "action_required"
    assert output["shipment_analysis"]["verdict"] == "seller_delay"
    assert output["shipment_analysis"]["late_seller_ids"] == [SELLER]
    assert output["payment_analysis"]["verdict"] == "reconciled"
    assert output["financial_resolution"]["recommended_refund_brl"] == 0
    assert output["affected_entities"]["order_ids"] == [ORDER]
    assert FAKE not in output["entity_resolution"]["resolved_order_ids"]
    assert FAKE in output["entity_resolution"]["rejected_candidates"]
    claims = {item["claim_id"]: item["verdict"] for item in output["claim_assessments"]}
    assert claims["L3B_CASE_010-1"] == "supported"
    assert claims["L3B_CASE_010-2"] == "unsupported"
    top_cause = output["root_cause_analysis"]["ranked_causes"][0]["cause_code"]
    assert top_cause == "SELLER_SHIPMENT_DELAY"
    assert output["root_cause_analysis"]["responsible_parties"][0]["party_id"] == SELLER
    assert output["evidence_refs"]
    assert "investigation_error" not in {event["event_type"] for event in events(trace)}


def test_order_delivered_and_shipment_lost_keeps_both_sources(trace: TraceWriter) -> None:
    gateway = gateway_for(
        order_status="delivered",
        shipment_payload={"shipment_id": "shp-lost", "shipment_status": "lost"},
        item_payload=items(price=40.0, freight=0.0),
        payment_rows=[
            {
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_value": 40.0,
                "payment_reference": "PAY-40",
            }
        ],
    )
    output = run(
        case("L3B_CASE_021", ["late_delivery_logistics", "requested_full_refund"]),
        gateway,
        trace,
    )

    validate(output)
    assert output["shipment_analysis"]["verdict"] == "lost"
    conflict = output["data_conflicts"][0]
    assert conflict["field"] == "shipment_status"
    assert conflict["sources"] == ["order", "shipment"]
    assert conflict["selected_source"] == "shipment"
    assert conflict["resolution_code"] == "SHIPMENT_SOURCE_PRECEDENCE"
    assert output["financial_resolution"]["recommended_refund_brl"] == 40
    assert output["assessment"]["case_status"] == "action_required"
    assert output["root_cause_analysis"]["ranked_causes"][0]["cause_code"] == "SHIPMENT_LOST"


def test_valid_split_payment_recommends_no_refund(trace: TraceWriter) -> None:
    gateway = gateway_for(
        payment_rows=[
            {
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_value": 40.0,
                "payment_reference": "P1",
            },
            {
                "payment_sequential": 2,
                "payment_type": "boleto",
                "payment_value": 60.0,
                "payment_reference": "P2",
            },
        ]
    )
    output = run(
        case("L3B_CASE_002", ["valid_split_payment", "requested_full_refund"]),
        gateway,
        trace,
    )

    validate(output)
    assert output["assessment"]["primary_issue"] == "valid_split_payment"
    assert output["assessment"]["case_status"] == "no_action"
    assert output["payment_analysis"]["verdict"] == "reconciled"
    assert output["payment_analysis"]["captured_total_brl"] == 100
    assert output["financial_resolution"]["recommended_refund_brl"] == 0
    assert output["financial_resolution"]["refund_lines"] == []
    assert output["resolution_actions"] == []
    assert output["shipment_analysis"]["verdict"] == "on_time"


def test_duplicate_capture_recommends_only_the_extra_charge(trace: TraceWriter) -> None:
    gateway = gateway_for(
        payment_rows=[
            {"payment_type": "credit_card", "payment_value": 50.0, "payment_reference": "D1"},
            {"payment_type": "credit_card", "payment_value": 50.0, "payment_reference": "D2"},
        ]
    )
    output = run(
        case("L3B_CASE_004", ["duplicate_charge", "requested_full_refund"]),
        gateway,
        trace,
    )

    validate(output)
    assert output["assessment"]["primary_issue"] == "duplicate_charge"
    assert output["payment_analysis"]["verdict"] == "duplicate_capture"
    assert output["financial_resolution"]["recommended_refund_brl"] == 50
    assert output["assessment"]["case_status"] == "action_required"
    assert output["resolution_actions"]


def test_canceled_paid_order_refunds_the_unrefunded_remainder(trace: TraceWriter) -> None:
    gateway = gateway_for(
        order_status="canceled",
        shipment_payload={"shipment_id": "shp-cancel", "shipment_status": "canceled"},
        payment_rows=[
            {
                "payment_sequential": 1,
                "payment_type": "credit_card",
                "payment_value": 90.0,
                "payment_reference": "C1",
            }
        ],
        item_payload=items(price=90.0, freight=0.0),
    )
    output = run(
        case("L3B_CASE_008", ["canceled_order_paid", "requested_full_refund"]),
        gateway,
        trace,
    )

    validate(output)
    assert output["assessment"]["primary_issue"] == "canceled_order_paid"
    assert output["financial_resolution"]["recommended_refund_brl"] == 90
    assert output["assessment"]["case_status"] == "action_required"
    reason = output["financial_resolution"]["refund_lines"][0]["reason_code"]
    assert reason == "CANCELED_ORDER_REFUND"


def test_verifier_repairs_overlap_seller_delay_and_invented_refund() -> None:
    entity = EntityResult(
        status="resolved",
        resolved_order_ids=["order-1"],
        rejected_candidates=["order-1", "order-2"],
        confidence=1.4,
        customer_unique_id="cust-1",
        evidence_refs=["not-a-ref", "ev_" + "a" * 32],
    )
    shipment_result = ShipmentResult(
        verdict="seller_delay",
        late_seller_ids=[],
        timeline_complete=True,
        seller_ids=["seller-9"],
        evidence_refs=["ev_" + "b" * 32],
    )
    payment = PaymentResult(
        verdict="refunded",
        captured_total_brl=10,
        refunded_total_brl=0,
        refundable_total_brl=10,
        financial_resolution=FinancialResolution(
            recommended_refund_brl=25,
            refund_lines=[RefundLine("X", 25, "order-1")],
        ),
        evidence_refs=["ev_" + "c" * 32],
    )
    output = verify_and_finalize(
        {
            "case_id": "L3B_CASE_050",
            "customer_request": {
                "claims": [{"claim_id": "claim-a", "topic": "late_delivery_seller"}]
            },
        },
        entity,
        shipment_result,
        payment,
    )

    validate(output)
    assert output["entity_resolution"]["resolved_order_ids"] == ["order-1"]
    assert output["entity_resolution"]["rejected_candidates"] == ["order-2"]
    assert output["entity_resolution"]["confidence"] == 1
    assert output["affected_entities"]["order_ids"] == ["order-1"]
    assert output["shipment_analysis"]["verdict"] == "seller_delay"
    assert output["shipment_analysis"]["late_seller_ids"] == ["seller-9"]
    assert output["financial_resolution"]["recommended_refund_brl"] == 0
    assert output["assessment"]["case_status"] == "action_required"
    assert output["resolution_actions"]
    assert "not-a-ref" not in output["evidence_refs"]
    assert 0 <= output["assessment"]["confidence"] <= 1
