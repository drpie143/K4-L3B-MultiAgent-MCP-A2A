from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from student_agent.agents.entity_agent import (
    entity_output_fields,
    extract_hints,
    extract_order_ids,
    find_order_record,
    resolve_entity,
)
from student_agent.contracts import Contracts
from student_agent.models.messages import EntityResult
from student_agent.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "L3B_CASE_001"
ORDER_A = "a" * 32
ORDER_B = "b" * 32
ORDER_C = "c" * 32
CUSTOMER = "f" * 32
OTHER_CUSTOMER = "e" * 32


def evidence(domain: str, data: Any, tag: str) -> dict[str, Any]:
    digest = hashlib.sha256(tag.encode()).hexdigest()
    return {
        "schema_version": "day09-mcp-evidence-v1",
        "evidence_ref": f"ev_{digest[:32]}",
        "result_hash": f"sha256:{digest}",
        "domain": domain,
        "data": data,
    }


def order_ev(order_id: str, customer: str = CUSTOMER, purchased: str = "2018-03-01") -> dict:
    row = {
        "order_id": order_id,
        "customer_id": f"cid-{order_id[:4]}",
        "customer_unique_id": customer,
        "order_status": "delivered",
        "order_purchase_timestamp": f"{purchased} 10:00:00",
    }
    return evidence("order", row, f"order-{order_id}-{customer}-{purchased}")


def history_ev(customer: str, order_ids: list[str]) -> dict:
    data = {"customer_unique_id": customer, "orders": [{"order_id": i} for i in order_ids]}
    return evidence("customer", data, f"history-{customer}-{'-'.join(order_ids)}")


class FakeGateway:
    """Scripted MCP gateway: responses keyed by (tool, main argument)."""

    def __init__(self, responses: dict[tuple[str, str], Any]) -> None:
        self.responses = {
            key: list(v) if isinstance(v, list) else [v] for key, v in responses.items()
        }
        self.calls: list[tuple[str, str, dict[str, str]]] = []

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        self.calls.append((tool_name, case_id, dict(arguments)))
        queue = self.responses.get((tool_name, next(iter(arguments.values()))))
        if not queue:
            raise RuntimeError(f"MCP tool {tool_name} failed: not found in case scope")
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def tools_called(self, tool_name: str) -> list[str]:
        return [next(iter(args.values())) for name, _, args in self.calls if name == tool_name]


@pytest.fixture
def trace(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", Contracts(ROOT / "contracts" / "schemas"))


def events(trace: TraceWriter) -> list[dict[str, Any]]:
    return [json.loads(line) for line in trace.path.read_text(encoding="utf-8").splitlines()]


def run(case: dict, gateway: FakeGateway, trace: TraceWriter, **kwargs: Any):
    return asyncio.run(resolve_entity(case, gateway, trace, **kwargs))


def all_gateway_refs(gateway: FakeGateway) -> set[str]:
    return {
        item["evidence_ref"]
        for queue in gateway.responses.values()
        for item in queue
        if isinstance(item, dict)
    }


def test_explicit_order_is_verified_and_customer_history_attached(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_order", ORDER_A): order_ev(ORDER_A),
            ("get_customer_history", CUSTOMER): history_ev(CUSTOMER, [ORDER_A, ORDER_B]),
        }
    )
    result = run({"case_id": CASE_ID, "order_id": ORDER_A}, gateway, trace)

    assert isinstance(result, EntityResult)  # shared A2A contract
    assert result.status == "resolved"
    assert result.resolved_order_ids == [ORDER_A]
    assert result.rejected_candidates == []
    assert result.customer_unique_id == CUSTOMER
    assert result.related_order_ids == [ORDER_B]
    assert result.confidence >= 0.9
    assert len(result.evidence_refs) == 2
    assert ORDER_A in result.order_evidence
    assert all(case_id == CASE_ID for _, case_id, _ in gateway.calls)


def test_does_not_default_to_first_candidate(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_order", ORDER_A): order_ev(ORDER_A, purchased="2018-01-10"),
            ("get_order", ORDER_B): order_ev(ORDER_B, purchased="2018-03-01"),
            ("get_order", ORDER_C): order_ev(ORDER_C, purchased="2018-05-20"),
        }
    )
    case = {
        "case_id": CASE_ID,
        "candidate_order_ids": [ORDER_A, ORDER_B, ORDER_C],
        "complaint": {"purchase_date": "2018-03-01"},
    }
    result = run(case, gateway, trace, fetch_history=False)

    assert result.status == "resolved"
    assert result.resolved_order_ids == [ORDER_B]
    assert set(result.rejected_candidates) == {ORDER_A, ORDER_C}
    assert not set(result.resolved_order_ids) & set(result.rejected_candidates)
    # Rejection-only refs are kept apart from supporting evidence.
    assert result.evidence_refs == [order_ev(ORDER_B, purchased="2018-03-01")["evidence_ref"]]
    assert len(result.rejection_evidence_refs) == 2


def test_indistinguishable_candidates_are_ambiguous(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_order", ORDER_A): order_ev(ORDER_A),
            ("get_order", ORDER_B): order_ev(ORDER_B),
        }
    )
    case = {"case_id": CASE_ID, "candidates": [{"order_id": ORDER_A}, {"order_id": ORDER_B}]}
    result = run(case, gateway, trace, fetch_history=False)

    assert result.status == "ambiguous"
    assert result.resolved_order_ids == []
    assert result.rejected_candidates == []
    assert result.confidence <= 0.5
    assert set(result.evidence_refs) == {
        order_ev(ORDER_A)["evidence_ref"],
        order_ev(ORDER_B)["evidence_ref"],
    }


def test_missing_candidates_are_not_found_without_invented_refs(trace: TraceWriter) -> None:
    gateway = FakeGateway({})
    case = {"case_id": CASE_ID, "candidate_order_ids": [ORDER_A, ORDER_B]}
    result = run(case, gateway, trace)

    assert result.status == "not_found"
    assert result.resolved_order_ids == []
    assert set(result.rejected_candidates) == {ORDER_A, ORDER_B}
    assert result.evidence_refs == []
    assert result.customer_unique_id is None
    assert result.candidate_decisions == {ORDER_A: "NOT_FOUND", ORDER_B: "NOT_FOUND"}


def test_customer_history_prunes_candidates_before_order_lookups(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_customer_history", CUSTOMER): history_ev(CUSTOMER, [ORDER_B, ORDER_C]),
            ("get_order", ORDER_B): order_ev(ORDER_B),
        }
    )
    case = {
        "case_id": CASE_ID,
        "customer": {"customer_unique_id": CUSTOMER},
        "candidate_order_ids": [ORDER_A, ORDER_B],
    }
    result = run(case, gateway, trace)

    assert result.status == "resolved"
    assert result.resolved_order_ids == [ORDER_B]
    assert result.rejected_candidates == [ORDER_A]
    assert result.candidate_decisions[ORDER_A] == "NOT_IN_CUSTOMER_HISTORY"
    assert gateway.tools_called("get_order") == [ORDER_B]
    assert gateway.tools_called("get_customer_history") == [CUSTOMER]
    assert result.related_order_ids == [ORDER_C]
    assert result.customer_unique_id == CUSTOMER


def test_customer_identifier_mismatch_rejects_candidate(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_order", ORDER_A): order_ev(ORDER_A, customer=OTHER_CUSTOMER),
            ("get_order", ORDER_B): order_ev(ORDER_B, customer=CUSTOMER),
            ("get_customer_history", CUSTOMER): history_ev(CUSTOMER, []),
        }
    )
    case = {
        "case_id": CASE_ID,
        "customer_unique_id": CUSTOMER,
        "candidate_order_ids": [ORDER_A, ORDER_B],
    }
    result = run(case, gateway, trace)

    assert result.resolved_order_ids == [ORDER_B]
    assert result.candidate_decisions[ORDER_A] == "CUSTOMER_MISMATCH"


def test_transport_failure_is_retried_once_and_tool_errors_are_not(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {("get_order", ORDER_A): [TimeoutError("timeout"), order_ev(ORDER_A, customer="")]}
    )
    result = run({"case_id": CASE_ID, "order_id": ORDER_A}, gateway, trace)
    assert result.status == "resolved"
    assert gateway.tools_called("get_order") == [ORDER_A, ORDER_A]

    failing = FakeGateway({("get_order", ORDER_B): TimeoutError("timeout")})
    result = run({"case_id": CASE_ID, "order_id": ORDER_B}, failing, trace)
    assert result.status == "not_found"
    assert len(failing.calls) == 2  # bounded: one retry only

    tool_error = FakeGateway({})
    run({"case_id": CASE_ID, "order_id": ORDER_C}, tool_error, trace)
    assert len(tool_error.calls) == 1


def test_undiscovered_tools_are_never_called(trace: TraceWriter) -> None:
    gateway = FakeGateway({("get_order", ORDER_A): order_ev(ORDER_A)})
    result = run(
        {"case_id": CASE_ID, "order_id": ORDER_A},
        gateway,
        trace,
        available_tools={"get_order"},
    )
    assert result.status == "resolved"
    assert gateway.tools_called("get_customer_history") == []
    assert result.related_order_ids == []


def test_case_without_any_order_reference_makes_no_calls(trace: TraceWriter) -> None:
    gateway = FakeGateway({})
    result = run({"case_id": CASE_ID, "message": "my package never arrived"}, gateway, trace)
    assert result.status == "not_found"
    assert gateway.calls == []


def test_invalid_case_id_is_rejected(trace: TraceWriter) -> None:
    with pytest.raises(ValueError, match="invalid case_id"):
        run({"case_id": "bad id"}, FakeGateway({}), trace)


def test_trace_is_observable_and_linked_to_evidence(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_order", ORDER_A): order_ev(ORDER_A),
            ("get_customer_history", CUSTOMER): history_ev(CUSTOMER, [ORDER_A]),
        }
    )
    result = run({"case_id": CASE_ID, "order_id": ORDER_A}, gateway, trace)
    emitted = events(trace)

    assert [e["event_type"] for e in emitted] == [
        "task_assigned",
        "tool_result_consumed",
        "tool_result_consumed",
        "handoff",
    ]
    assert all(e["case_id"] == CASE_ID for e in emitted)
    assert emitted[0]["target"] == "entity-agent"
    assert emitted[-1]["actor"] == "entity-agent" and emitted[-1]["target"] == "coordinator"
    consumed = {ref for e in emitted[1:-1] for ref in e["evidence_refs"]}
    assert set(result.evidence_refs) <= consumed <= all_gateway_refs(gateway)
    assert not any("reason" in key for e in emitted for key in e.get("attributes", {}))


def test_output_fields_fit_the_l3b_schema(trace: TraceWriter) -> None:
    gateway = FakeGateway(
        {
            ("get_order", ORDER_A): order_ev(ORDER_A),
            ("get_customer_history", CUSTOMER): history_ev(CUSTOMER, [ORDER_A, ORDER_B]),
        }
    )
    fields = entity_output_fields(run({"case_id": CASE_ID, "order_id": ORDER_A}, gateway, trace))
    output = {
        "schema_version": "day09-l3b-output-v2",
        "case_id": CASE_ID,
        "assessment": {
            "primary_issue": "insufficient_evidence",
            "secondary_issues": [],
            "case_status": "needs_investigation",
            "confidence": 0.5,
        },
        "affected_entities": {
            "order_ids": fields["order_ids"],
            "item_ids": [],
            "seller_ids": [],
            "payment_references": [],
            "shipment_ids": [],
        },
        "entity_resolution": fields["entity_resolution"],
        "customer_context": fields["customer_context"],
        "shipment_analysis": {
            "verdict": "insufficient_evidence",
            "late_seller_ids": [],
            "timeline_complete": False,
        },
        "payment_analysis": {
            "verdict": "insufficient_evidence",
            "captured_total_brl": None,
            "refunded_total_brl": None,
            "refundable_total_brl": None,
        },
        "root_cause_analysis": {"ranked_causes": [], "responsible_parties": []},
        "evidence_refs": fields["evidence_refs"],
        "data_conflicts": [],
        "financial_resolution": {
            "currency": "BRL",
            "recommended_refund_brl": 0,
            "refund_lines": [],
        },
        "resolution_actions": [],
    }
    Contracts(ROOT / "contracts" / "schemas").validate_output(output, "entity fragment")


def test_case_parsing_helpers() -> None:
    case = {
        "case_id": CASE_ID,
        "order_id": ORDER_A,
        "candidate_orders": [{"order_id": ORDER_B}, ORDER_C, ORDER_B],
        "customer": {"customer_unique_id": CUSTOMER, "customer_state": "SP"},
        "history": [{"order_id": "ignored-inside-list"}],
    }
    claimed, candidates, text_ids = extract_order_ids(case)
    assert claimed == [ORDER_A]
    assert candidates == [ORDER_B, ORDER_C]
    assert text_ids == []
    assert extract_hints(case) == {"customer_unique_id": CUSTOMER, "customer_state": "SP"}

    assert find_order_record({"order": {"order_id": ORDER_A}}, ORDER_A) == {"order_id": ORDER_A}
    assert find_order_record({"order_id": ORDER_B}, ORDER_A) is None
    assert find_order_record({"found": False}, ORDER_A) is None
    assert find_order_record(None, ORDER_A) is None


def test_case_level_customer_hint_prunes_fake_candidate_without_order_lookup(
    trace: TraceWriter,
) -> None:
    hint = "customer-5358baaa785d"
    gateway = FakeGateway(
        {
            ("get_customer_history", hint): history_ev(hint, [ORDER_A, ORDER_A]),
            ("get_order", ORDER_A): evidence(
                "order", {"order_id": ORDER_A, "customer_id": "customer-row-x"}, "row"
            ),
        }
    )
    case = {
        "case_id": CASE_ID,
        "customer_request": {"claimed_order_id": ORDER_A},
        "candidate_order_ids": [ORDER_A, "candidate-004"],
        "customer_unique_id_hint": hint,
    }
    result = run(case, gateway, trace)

    assert result.status == "resolved"
    assert result.resolved_order_ids == [ORDER_A]
    assert result.rejected_candidates == ["candidate-004"]
    assert result.customer_unique_id == hint
    assert result.related_order_ids == []
    assert gateway.tools_called("get_order") == [ORDER_A]


def test_wrong_claimed_order_is_recovered_from_customer_history(trace: TraceWriter) -> None:
    hint = "customer-000000000001"
    gateway = FakeGateway(
        {
            ("get_customer_history", hint): history_ev(hint, [ORDER_B]),
            ("get_order", ORDER_B): evidence("order", {"order_id": ORDER_B}, "real-order"),
        }
    )
    case = {
        "case_id": CASE_ID,
        "customer_request": {"claimed_order_id": ORDER_A},
        "candidate_order_ids": [ORDER_A, "candidate-001"],
        "customer_unique_id_hint": hint,
    }
    result = run(case, gateway, trace)

    assert result.status == "resolved"
    assert result.resolved_order_ids == [ORDER_B]
    assert set(result.rejected_candidates) == {ORDER_A, "candidate-001"}
    assert gateway.tools_called("get_order") == [ORDER_B]
    assert evidence("order", {}, "real-order")["evidence_ref"] in result.evidence_refs
