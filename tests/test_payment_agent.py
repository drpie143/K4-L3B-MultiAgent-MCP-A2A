from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from student_agent.agents.payment_agent import investigate_payment
from student_agent.contracts import Contracts
from student_agent.models.messages import EntityResult
from student_agent.trace import TraceWriter


class MockGateway:
    """Mock MCP Gateway simulating tools and responses."""

    def __init__(self, tools: list[str], responses: dict[str, dict[str, Any]]) -> None:
        self._tools = tools
        self._responses = responses
        self.call_history: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[str]:
        return list(self._tools)

    async def call(self, tool_name: str, *, case_id: str, **kwargs: Any) -> dict[str, Any]:
        self.call_history.append((tool_name, {"case_id": case_id, **kwargs}))
        if tool_name in self._responses:
            return self._responses[tool_name]
        raise ValueError(f"Unknown mock tool: {tool_name}")


@pytest.fixture
def contracts() -> Contracts:
    root = Path(__file__).resolve().parents[1]
    return Contracts(root / "contracts" / "schemas")


@pytest.fixture
def trace(tmp_path: Path, contracts: Contracts) -> TraceWriter:
    trace_path = tmp_path / "traces" / "trace.jsonl"
    return TraceWriter(trace_path, contracts)


def test_fallback_when_entity_unresolved(trace: TraceWriter) -> None:
    """Test that when entity status is not resolved, payment agent does NOT call MCP."""

    async def _run() -> None:
        gateway = MockGateway(tools=["get_payment_details"], responses={})
        case = {"case_id": "L3B_CASE_001"}

        entity_not_found = EntityResult(
            status="not_found",
            resolved_order_ids=[],
            rejected_candidates=["ORD_999"],
            confidence=0.5,
        )

        result = await investigate_payment(case, entity_not_found, gateway, trace)  # type: ignore[arg-type]

        assert result.verdict == "insufficient_evidence"
        assert result.captured_total_brl is None
        assert result.financial_resolution.recommended_refund_brl == 0.0
        assert result.financial_resolution.refund_lines == []
        # Assert zero calls to MCP gateway (saves budget!)
        assert len(gateway.call_history) == 0

    asyncio.run(_run())


def test_reconciled_payment(trace: TraceWriter) -> None:
    """Test standard single or split payment reconciled cleanly."""

    async def _run() -> None:
        mock_payment_response = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_0123456789abcdef0123456789abcdef",
            "result_hash": "sha256:" + "0" * 64,
            "domain": "payment",
            "data": [
                {
                    "payment_reference": "TXN_001",
                    "payment_sequential": 1,
                    "payment_type": "credit_card",
                    "payment_value": 150.50,
                    "status": "captured",
                }
            ],
        }

        gateway = MockGateway(
            tools=["get_payment_details"],
            responses={"get_payment_details": mock_payment_response},
        )

        case = {"case_id": "L3B_CASE_002"}
        entity = EntityResult(
            status="resolved",
            resolved_order_ids=["ORD_123"],
            confidence=1.0,
        )

        result = await investigate_payment(case, entity, gateway, trace)  # type: ignore[arg-type]

        assert result.verdict == "reconciled"
        assert result.captured_total_brl == 150.50
        assert result.refunded_total_brl == 0.0
        assert result.refundable_total_brl == 150.50
        assert result.payment_references == ["TXN_001"]
        assert result.financial_resolution.recommended_refund_brl == 0.0
        assert result.financial_resolution.currency == "BRL"
        assert "ev_0123456789abcdef0123456789abcdef" in result.evidence_refs
        assert len(gateway.call_history) == 1

    asyncio.run(_run())


def test_duplicate_capture_detection(trace: TraceWriter) -> None:
    """Test duplicate transaction detection and recommended refund calculation."""

    async def _run() -> None:
        mock_payment_response = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_duplicate_1234567890abcdef12345678",
            "result_hash": "sha256:" + "1" * 64,
            "domain": "payment",
            "data": [
                {
                    "payment_reference": "TXN_001",
                    "payment_type": "credit_card",
                    "payment_value": 85.00,
                    "status": "captured",
                },
                {
                    "payment_reference": "TXN_002",
                    "payment_type": "credit_card",
                    "payment_value": 85.00,
                    "status": "captured",
                },
            ],
        }

        gateway = MockGateway(
            tools=["get_payment_details"],
            responses={"get_payment_details": mock_payment_response},
        )

        case = {"case_id": "L3B_CASE_003"}
        entity = EntityResult(
            status="resolved",
            resolved_order_ids=["ORD_456"],
            confidence=1.0,
        )

        result = await investigate_payment(case, entity, gateway, trace)  # type: ignore[arg-type]

        assert result.verdict == "duplicate_capture"
        assert result.captured_total_brl == 170.00
        assert result.financial_resolution.recommended_refund_brl == 85.00
        assert len(result.financial_resolution.refund_lines) == 1
        assert result.financial_resolution.refund_lines[0].reason_code == "DUPLICATE_CHARGE_REFUND"
        assert result.financial_resolution.refund_lines[0].amount_brl == 85.00
        assert result.candidate_causes[0].cause_code == "DUPLICATE_CHARGE"
        assert result.candidate_causes[0].party_type == "payment_provider"

    asyncio.run(_run())


def test_refund_pending_and_failed(trace: TraceWriter) -> None:
    """Test refund pending and refund failed scenarios."""

    async def _run() -> None:
        mock_payment_response = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_payment_999999999999999999999999",
            "result_hash": "sha256:" + "2" * 64,
            "domain": "payment",
            "data": [
                {"payment_reference": "TXN_999", "payment_value": 200.0, "status": "captured"}
            ],
        }
        mock_refund_failed_response = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_refund_8888888888888888888888888",
            "result_hash": "sha256:" + "3" * 64,
            "domain": "refund",
            "data": [{"refund_id": "REF_001", "refund_amount": 200.0, "refund_status": "failed"}],
        }

        gateway = MockGateway(
            tools=["get_payment_details", "get_refund_status"],
            responses={
                "get_payment_details": mock_payment_response,
                "get_refund_status": mock_refund_failed_response,
            },
        )

        case = {"case_id": "L3B_CASE_004"}
        entity = EntityResult(
            status="resolved",
            resolved_order_ids=["ORD_789"],
            confidence=1.0,
        )

        result = await investigate_payment(case, entity, gateway, trace)  # type: ignore[arg-type]

        assert result.verdict == "refund_failed"
        assert result.financial_resolution.recommended_refund_brl == 200.0
        assert result.candidate_causes[0].cause_code == "REFUND_GATEWAY_FAILURE"

    asyncio.run(_run())


def test_capture_mismatch(trace: TraceWriter) -> None:
    """Test payment mismatch when captured total differs from expected total."""

    async def _run() -> None:
        mock_payment_response = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_payment_mismatch_1234567890abcdef",
            "result_hash": "sha256:" + "4" * 64,
            "domain": "payment",
            "data": [
                {"payment_reference": "TXN_MISMATCH", "payment_value": 150.0, "status": "captured"}
            ],
        }

        gateway = MockGateway(
            tools=["get_payment_details"],
            responses={"get_payment_details": mock_payment_response},
        )

        case = {
            "case_id": "L3B_CASE_005",
            "expected_total_brl": 100.0,  # Expected 100, but captured 150 -> overcharged 50
        }
        entity = EntityResult(
            status="resolved",
            resolved_order_ids=["ORD_555"],
            confidence=1.0,
        )

        result = await investigate_payment(case, entity, gateway, trace)  # type: ignore[arg-type]

        assert result.verdict == "capture_mismatch"
        assert result.captured_total_brl == 150.0
        assert result.financial_resolution.recommended_refund_brl == 50.0
        assert len(result.financial_resolution.refund_lines) == 1
        assert result.financial_resolution.refund_lines[0].reason_code == "OVERCHARGE_REFUND"

    asyncio.run(_run())


def test_payment_output_strictly_passes_contract_schema(
    trace: TraceWriter, contracts: Contracts
) -> None:
    """Validate that payment_analysis and financial_resolution strictly satisfy the L3B schema."""

    async def _run() -> None:
        mock_payment_response = {
            "schema_version": "day09-mcp-evidence-v1",
            "evidence_ref": "ev_valid_contract_123456789012345678",
            "result_hash": "sha256:" + "5" * 64,
            "domain": "payment",
            "data": [
                {
                    "payment_reference": "TXN_CONTRACT_001",
                    "payment_value": 99.90,
                    "status": "captured",
                }
            ],
        }

        gateway = MockGateway(
            tools=["get_payment_details"],
            responses={"get_payment_details": mock_payment_response},
        )

        case = {"case_id": "L3B_CASE_006"}
        entity = EntityResult(
            status="resolved",
            resolved_order_ids=["ORD_006"],
            confidence=1.0,
        )

        payment_res = await investigate_payment(case, entity, gateway, trace)  # type: ignore[arg-type]

        # Construct a synthetic full output with payment_analysis and financial_resolution
        full_output = {
            "schema_version": "day09-l3b-output-v2",
            "case_id": "L3B_CASE_006",
            "assessment": {
                "primary_issue": "valid_split_payment",
                "secondary_issues": [],
                "case_status": "no_action",
                "confidence": 0.95,
            },
            "affected_entities": {
                "order_ids": ["ORD_006"],
                "item_ids": [],
                "seller_ids": [],
                "payment_references": payment_res.payment_references,
                "shipment_ids": [],
            },
            "entity_resolution": {
                "status": "resolved",
                "resolved_order_ids": ["ORD_006"],
                "rejected_candidates": [],
                "confidence": 1.0,
            },
            "customer_context": {
                "customer_unique_id": None,
                "related_order_ids": [],
            },
            "shipment_analysis": {
                "verdict": "on_time",
                "late_seller_ids": [],
                "timeline_complete": True,
            },
            "payment_analysis": payment_res.to_dict(),
            "root_cause_analysis": {
                "ranked_causes": [{"cause_code": "CUSTOMER_CONFUSION", "rank": 1}],
                "responsible_parties": [{"party_type": "customer", "party_id": None}],
            },
            "evidence_refs": payment_res.evidence_refs,
            "data_conflicts": [],
            "financial_resolution": payment_res.financial_resolution.to_dict(),
            "resolution_actions": [],
        }

        # Must pass schema validation with zero errors!
        contracts.validate_output(full_output, "test_payment_contract_validation")

    asyncio.run(_run())
