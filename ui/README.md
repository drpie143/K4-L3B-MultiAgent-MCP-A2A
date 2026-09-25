# L3B workflow viewer

Giao diện xem luồng multi-agent theo từng case: input, pipeline Entity → Coordinator →
Order/Shipment ∥ Payment/Refund → Conflict → Verifier → Output, kết quả từng agent,
verifier checklist và trace timeline.

```bash
python ui/server.py            # http://127.0.0.1:8765
python ui/server.py --port 9000
```

Hai chế độ xem:

- **Hội thoại**: trace của case hiển thị như một cuộc chat A2A (khách → Coordinator → các agent →
  kết luận cuối), có nút "▶ Phát lại" để hiện từng tin một khi thuyết trình.
- **Dashboard**: sơ đồ luồng, kết quả từng agent, verifier checklist, trace timeline, JSON thô.

- Chỉ đọc `case-set.json`, `inputs/`, `outputs/`, `traces/trace.jsonl`; không gọi MCP, không ghi file.
- Đọc lại file mỗi request: bật "Tự làm mới 3s" trong lúc chạy `day09 run` để xem tiến độ.
- "Dữ liệu mẫu" chỉ giả lập trong trình duyệt để xem giao diện, không phải kết quả thật.
- Tên actor trong trace được map vào từng bước qua mảng `STAGES` trong `index.html`.
  Khi agent của Người 2/3/4 dùng tên actor khác, thêm alias vào `actors` của bước tương ứng.
