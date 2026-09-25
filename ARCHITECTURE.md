# L3B Architecture Record

Coordinator cho L3B. Tài liệu này ghi quyết định kiểm chứng được. Không ghi prompt, chain-of-thought hay API key.

## 1. System overview

```text
case
  → entity-agent
  → nếu chưa resolve: verifier kết luận insufficient_evidence
  → nếu đã resolve:
        order-shipment-agent  ┐
        payment-agent         ┘  (asyncio.gather)
  → verifier (conflict + consistency)
  → output day09-l3b-output-v2
```

`day09 run` ghi `case_received` trước `solve_case` và `case_finalized` sau khi output hợp schema. Trong `solve_case`, từng agent ghi `task_assigned`, `tool_result_consumed`, `handoff`. Verifier ghi `verification_completed`.

Trace chỉ chứa mã sự kiện và thuộc tính quan sát được (`verdict`, `status`, số lượng, `case_id`). Không ghi câu suy luận.

## 2. Agent ownership

| Actor | Input | Trách nhiệm | Tool được gọi | Output |
| --- | --- | --- | --- | --- |
| entity-agent | raw case | Chọn order bằng evidence, loại candidate sai, lấy `customer_unique_id` | `get_order`, `get_customer_history` | `EntityResult` |
| order-shipment-agent | `EntityResult` | Item, seller, timeline giao hàng, candidate cause | `get_order_items`, `get_shipment_summary`; `get_product_context` chỉ khi case bật cờ | `OrderShipmentResult` |
| payment-agent | `EntityResult` | Capture, split, duplicate, refund | `get_payment_timeline` hoặc `get_order_payments`, `get_refund_timeline` | `PaymentResult` |
| coordinator | case | Gọi entity trước, specialist song song sau khi resolve, lấy policy của case | `list_tools`, `get_policy` | handoff sang verifier, `policy_decided` |
| verifier | ba result + policy | Đối chiếu tín hiệu evidence với khiếu nại, áp rule policy, ghi conflict, dựng output | không gọi MCP | dict đúng schema |

Contract nội bộ là `src/student_agent/models/messages.py`: `EntityResult`, `ShipmentResult`, `PaymentResult`, `CandidateCause`, `FinancialResolution`.

`customer_unique_id_hint` là khóa của `get_customer_history`. Entity gọi history trước; danh sách order authoritative trong history loại candidate giả mà không cần `get_order` cho nó. Nếu mọi candidate đều sai, order duy nhất trong history được kiểm bằng `get_order` rồi mới resolve. Order row không mang `customer_unique_id`, nên hint không bị so khớp cứng với order row.

## 3. Entity resolution và A2A protocol

Entity agent giữ luật: không lấy candidate đầu tiên, không resolve khi evidence chưa tách được một order. `ambiguous` hoặc `not_found` thì coordinator không gọi specialist và không đề xuất refund.

Handoff đi theo `case_id`. Không có vòng lặp agent. Mỗi specialist chạy một lần. Lỗi specialist được nuốt thành `insufficient_evidence` kèm `decision_code` `SHIPMENT_FAILED` hoặc `PAYMENT_FAILED`.

## 4. Evidence và conflict lifecycle

Mọi `gateway.call` truyền `case_id` của case đang xử lý. `evidence_ref` chỉ lấy từ envelope MCP, lọc bằng pattern `ev_...` trước khi đưa vào output hoặc trace.

Cache nằm ở `utils/cache.py`, khóa `(case_id, tool, args)`. Miss cũng được nhớ trong phạm vi case. Không dùng lại evidence của case khác.

Khi order nói delivered và shipment nói lost hoặc returned, verifier ghi:

```json
{
  "field": "shipment_status",
  "sources": ["order", "shipment"],
  "selected_source": "shipment",
  "resolution_code": "SHIPMENT_SOURCE_PRECEDENCE"
}
```

Khi tổng item và tổng capture lệch hơn 1 BRL:

```json
{
  "field": "captured_total_brl",
  "sources": ["order_items", "payment"],
  "selected_source": "payment",
  "resolution_code": "PAYMENT_CAPTURE_PRECEDENCE"
}
```

Không chọn nguồn trong im lặng. Conflict không phân được thì `selected_source` là null và `resolution_code` là `UNRESOLVED_CONFLICT`.

Khi có policy (luồng chính), verifier dùng `agents/case_signals.py`:

- Dữ liệu một order trộn hai mốc thời gian. Dòng có ngày sau `opened_at` quá 14 ngày không giải thích được khiếu nại nên bị bỏ khi phát hiện vấn đề.
- Mỗi phiên bản order (order row, các dòng history) chỉ so với shipping limit của chính nó (limit nằm trong 30 ngày sau ngày mua).
- Duplicate charge chỉ khi có dòng payment giống hệt nhau (cùng sequential, type, value); split credit + voucher không phải duplicate.
- Khiếu nại được evidence xác nhận thì thành primary issue (confidence 0.92). Khiếu nại không được xác nhận nhường cho tín hiệu mạnh (refund failed/pending, duplicate, canceled/unavailable đã thu tiền, seller trễ). Không có tín hiệu nào thì giữ khiếu nại với confidence thấp.
- `case_status`, `recommended_refund_brl`, `resolution_actions` và `responsible_parties` lấy từ rule policy của primary issue. `party_id` seller mẫu trong policy được thay bằng seller có evidence trễ hạn.
- Order row và history mâu thuẫn trạng thái thì ghi `data_conflicts` với `resolution_code` `COMPLAINT_TIMELINE_PRECEDENCE`.

Không có policy thì dùng thứ tự primary issue cũ khi nhiều tín hiệu cùng đúng: duplicate capture, lệch tiền, refund failed, refund pending, đơn canceled đã thu tiền, đơn unavailable đã thu tiền, seller giao trễ, logistics trễ hoặc lost/returned, split payment hợp lệ. Không còn tín hiệu nào và cả shipment lẫn payment đều có evidence thì `unsupported_claim`.

Refund chỉ xuất hiện khi verdict tài chính hoặc hàng không giao được yêu cầu hoàn tiền. Tiền đã capture không bị đổi thành refund. Split có `payment_sequential` khác nhau không bị coi là duplicate. Đơn trễ nhưng đã giao không tự động full refund.

## 5. Failure and efficiency policy

| Failure | Retry | Fallback | Trace |
| --- | ---: | --- | --- |
| MCP timeout / lỗi mạng | 2 | bỏ payload đó | không bịa ref |
| Tool trả lỗi nghiệp vụ | 0 | coi như không có evidence | không retry |
| Entity not_found / ambiguous | 0 | `insufficient_evidence`, refund 0 | `verification_completed` |
| Specialist raise | 0 | verdict insufficient | `SHIPMENT_FAILED` / `PAYMENT_FAILED` |
| Conflict đã chọn được nguồn | 0 | ghi `data_conflicts` | `conflict_count` |
| Conflict chưa chọn được | 0 | `needs_investigation` | `selected_source: null` |

Mỗi case dùng 8 call: `get_customer_history`, `get_order` (order thật), `get_order_items`, `get_shipment_summary`, `get_product_context`, `get_payment_timeline`, `get_refund_timeline`, `get_policy`. Evidence được lưu ở `.mcp_store/<case_id>/` (`utils/evidence_store.py`, gitignore): chạy lại `day09 run` phát lại evidence đã lưu, giữ nguyên `evidence_ref`, không tạo call mới. Tắt bằng `DAY09_EVIDENCE_STORE=off`. Lỗi nghiệp vụ của tool cũng được lưu để không tra lại candidate sai; lỗi mạng thì không.

Không gọi `get_sellers` khi item đã có `seller_id`. Không gọi lại `get_order` nếu entity đã cache envelope. `get_product_context` chỉ chạy khi `investigation_scope.include_product_context` là true. `get_order_payments` chỉ là fallback khi payment timeline không có dòng `payment_value`.

## 6. Verification invariants

Trước khi trả output, verifier ép các bất biến sau:

1. `case_id` lấy từ case, không suy ra id khác.
2. `resolved_order_ids` và `rejected_candidates` không giao nhau.
3. `affected_entities.order_ids` bằng `resolved_order_ids`.
4. `seller_delay` thì `late_seller_ids` không rỗng; không có seller id thì hạ verdict xuống `insufficient_evidence`.
5. `refunded` mà `refunded_total_brl` không dương thì không giữ verdict đó.
6. `recommended_refund_brl` không vượt `refundable_total_brl`.
7. `action_required` thì `resolution_actions` không rỗng.
8. `no_action` thì refund đề xuất bằng 0 và không có action.
9. `evidence_refs` đúng pattern và không vượt 30 phần tử.
10. Mọi confidence nằm trong `[0, 1]`.
11. Cause đứng đầu khớp shipment verdict (`SHIPMENT_LOST`, `SELLER_SHIPMENT_DELAY`, ...) hoặc payment verdict.
12. Conflict có ít nhất hai source. Source được chọn phải thuộc danh sách đó, hoặc null nếu chưa chọn.

## 7. Reproducibility

- Python `>=3.11`, dependency trong `pyproject.toml`.
- Không có random seed. Hai specialist chạy bằng `asyncio.gather`; trace ghi đồng bộ trên event loop nên mỗi dòng JSONL là một event trọn vẹn.
- Lệnh kiểm tra: `pytest -q`. Lệnh chạy submission: `day09 run` rồi `day09 validate`.
- MCP tool được discovery từ server, không hard-code case id.
- Team API key chỉ đọc từ môi trường lúc chạy. Không ghi vào output, trace hay git.
