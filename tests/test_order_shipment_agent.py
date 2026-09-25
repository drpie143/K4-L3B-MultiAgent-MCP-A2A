"""Order and shipment specialist."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from student_agent.agents.order_shipment_agent import investigate_order_shipment
from student_agent.contracts import Contracts
from student_agent.models.messages import EntityResult
from student_agent.trace import TraceWriter

ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "L3B_CASE_010"
ORDER = "b" * 32


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
    def __init__(self, responses: dict[tuple[str, str], dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    async def list_tools(self) -> list[str]:
        return sorted({tool for tool, _ in self.responses})

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        assert case_id == CASE_ID
        self.calls.append(tool_name)
        return self.responses[(tool_name, next(iter(arguments.values())))]


def trace_writer(tmp_path: Path) -> TraceWriter:
    return TraceWriter(tmp_path / "trace.jsonl", Contracts(ROOT / "contracts" / "schemas"))


def test_missing_shipment_evidence_is_not_on_time(tmp_path: Path) -> None:
    gateway = Gateway({})
    entity = EntityResult(status="resolved", resolved_order_ids=[ORDER], confidence=0.8)
    result = asyncio.run(
        investigate_order_shipment({"case_id": CASE_ID}, entity, gateway, trace_writer(tmp_path))  # type: ignore[arg-type]
    )

    assert result.verdict == "insufficient_evidence"
    assert result.timeline_complete is False
    assert result.evidence_refs == []
    events = [
        json.loads(line)
        for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {event["event_type"] for event in events} <= {
        "task_assigned",
        "tool_result_consumed",
        "handoff",
        "verification_completed",
        "case_received",
        "case_finalized",
        "policy_decided",
    }


def test_seller_handoff_after_limit_names_the_seller(tmp_path: Path) -> None:
    gateway = Gateway(
        {
            ("get_order_items", ORDER): evidence(
                "item",
                {
                    "items": [
                        {
                            "order_item_id": "item-9",
                            "seller_id": "seller-9",
                            "price": 10,
                            "freight_value": 1,
                            "shipping_limit_date": "2018-01-05 00:00:00",
                        }
                    ]
                },
                "items",
            ),
            ("get_shipment_summary", ORDER): evidence(
                "shipment",
                {
                    "shipment_id": "shp-9",
                    "shipment_status": "delivered",
                    "order_delivered_carrier_date": "2018-01-12 00:00:00",
                    "order_delivered_customer_date": "2018-01-18 00:00:00",
                    "order_estimated_delivery_date": "2018-01-20 00:00:00",
                },
                "shipment",
            ),
        }
    )
    entity = EntityResult(status="resolved", resolved_order_ids=[ORDER], confidence=0.9)
    result = asyncio.run(
        investigate_order_shipment(
            {"case_id": CASE_ID, "investigation_scope": {"include_product_context": False}},
            entity,
            gateway,  # type: ignore[arg-type]
            trace_writer(tmp_path),
        )
    )

    assert result.verdict == "seller_delay"
    assert result.late_seller_ids == ["seller-9"]
    assert result.item_ids == ["item-9"]
    assert result.shipment_ids == ["shp-9"]
    assert result.timeline_complete is True
    assert result.order_total_brl == 11
    assert result.candidate_causes[0].cause_code == "SELLER_SHIPMENT_DELAY"
    assert gateway.calls == ["get_shipment_summary", "get_order_items"]
