from __future__ import annotations

import asyncio
from typing import Any

from src.student_agent.messages import (
    CandidateCause,
    EntityResult,
    ShipmentResult,
    ShipmentVerdict,
)
from src.student_agent.mcp_gateway import EvidenceGateway
from src.student_agent.trace import TraceWriter


async def investigate_order_shipment(
    case: dict[str, Any],
    entity: EntityResult,
    gateway: EvidenceGateway,
    trace: TraceWriter,
) -> ShipmentResult:
    """
    Nhiệm vụ: Người số 2 - Điều tra Order/Product + Shipment
    """
    # Ghi nhận log: Agent bắt đầu làm việc
    trace.emit(
        case_id=case["case_id"],
        event_type="task_assigned",
        actor="order_shipment_agent",
        attributes={"status": "started"},
    )

    # Nếu không có order nào được resolve từ Person 1, trả về insufficient_evidence
    if entity.status != "resolved" or not entity.resolved_order_ids:
        return ShipmentResult(
            verdict="insufficient_evidence",
            timeline_complete=False,
        )

    all_item_ids: list[str] = []
    all_seller_ids: list[str] = []
    all_shipment_ids: list[str] = []
    all_evidence_refs: list[str] = []
    late_seller_ids: list[str] = []
    candidate_causes: list[CandidateCause] = []

    # Giả định: Có thể gọi nhiều tool để lấy thông tin, ví dụ "get_order_details", "get_shipment_details"
    # (Bạn cần dùng day09 mcp-tools để xem chính xác tên tool do BTC cung cấp)
    
    # Ở đây chúng ta sẽ đi qua từng order đã được Entity Agent xác nhận
    for order_id in entity.resolved_order_ids:
        try:
            # 1. Gọi MCP Tool để lấy chi tiết order
            order_evidence = await gateway.call(
                "get_order", # TODO: Đổi thành tên tool chính xác từ MCP
                case_id=case["case_id"],
                order_id=order_id,
            )
            
            evidence_ref = order_evidence["evidence_ref"]
            all_evidence_refs.append(evidence_ref)
            order_data = order_evidence["data"]
            
            trace.emit(
                case_id=case["case_id"],
                event_type="tool_result_consumed",
                actor="order_shipment_agent",
                tool_name="get_order",
                evidence_refs=[evidence_ref],
            )

            # Thu thập items và sellers
            items = order_data.get("items", [])
            for item in items:
                if "item_id" in item:
                    all_item_ids.append(item["item_id"])
                if "seller_id" in item:
                    all_seller_ids.append(item["seller_id"])

            # 2. Gọi MCP Tool lấy thông tin vận chuyển
            shipment_evidence = await gateway.call(
                "get_shipment", # TODO: Đổi thành tên tool chính xác từ MCP
                case_id=case["case_id"],
                order_id=order_id,
            )
            
            ship_ev_ref = shipment_evidence["evidence_ref"]
            all_evidence_refs.append(ship_ev_ref)
            shipment_data = shipment_evidence["data"]
            
            trace.emit(
                case_id=case["case_id"],
                event_type="tool_result_consumed",
                actor="order_shipment_agent",
                tool_name="get_shipment",
                evidence_refs=[ship_ev_ref],
            )
            
            if "shipment_id" in shipment_data:
                all_shipment_ids.append(shipment_data["shipment_id"])

            # Tự phân tích (Shipment analysis logic)
            # Dưới đây là logic giả lập, bạn cần viết logic kiểm tra ngày giao hàng thực tế vs dự kiến
            status = shipment_data.get("status")
            if status == "lost":
                return ShipmentResult(
                    verdict="lost",
                    item_ids=list(set(all_item_ids)),
                    seller_ids=list(set(all_seller_ids)),
                    shipment_ids=list(set(all_shipment_ids)),
                    evidence_refs=all_evidence_refs,
                )
            elif status == "returned":
                return ShipmentResult(
                    verdict="returned",
                    item_ids=list(set(all_item_ids)),
                    seller_ids=list(set(all_seller_ids)),
                    shipment_ids=list(set(all_shipment_ids)),
                    evidence_refs=all_evidence_refs,
                )
            
            # Logic kiểm tra trễ hạn (cần kiểm tra date)
            # Ví dụ: nếu có thông tin seller giao trễ
            # late_seller_ids.append(...)

        except Exception as e:
            # Nếu MCP call lỗi
            trace.emit(
                case_id=case["case_id"],
                event_type="investigation_error",
                actor="order_shipment_agent",
                attributes={"error": str(e)},
            )
            pass

    # Xử lý tổng hợp (Placeholder)
    verdict: ShipmentVerdict = "on_time"
    timeline_complete = True
    
    if late_seller_ids:
        verdict = "seller_delay"
        candidate_causes.append(CandidateCause(
            cause_code="SELLER_SHIPMENT_DELAY",
            party_type="seller",
            party_id=late_seller_ids[0]
        ))
        
    # Ghi nhận handoff trước khi kết thúc
    trace.emit(
        case_id=case["case_id"],
        event_type="handoff",
        actor="order_shipment_agent",
        target="coordinator",
    )

    return ShipmentResult(
        verdict=verdict,
        late_seller_ids=list(set(late_seller_ids)),
        timeline_complete=timeline_complete,
        item_ids=list(set(all_item_ids)),
        seller_ids=list(set(all_seller_ids)),
        shipment_ids=list(set(all_shipment_ids)),
        candidate_causes=candidate_causes,
        evidence_refs=all_evidence_refs,
    )
