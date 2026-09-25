"""Local read-only viewer for L3B inputs, outputs and trace.

Usage:
    python ui/server.py [--root .] [--port 8765]

Binds to 127.0.0.1 only. Reads files on every request so a running ``day09 run``
shows up live. Never calls MCP and never writes anything.
"""

from __future__ import annotations

import argparse
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

CASE_ID = re.compile(r"^[A-Z0-9][A-Z0-9_-]{2,63}$")
UI_DIR = Path(__file__).resolve().parent


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_trace(root: Path) -> list[dict[str, Any]]:
    try:
        lines = (root / "traces" / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def summary(root: Path) -> dict[str, Any]:
    case_set = _read_json(root / "case-set.json") or {}
    case_ids = case_set.get("case_ids") or sorted(p.stem for p in (root / "inputs").glob("*.json"))
    trace = _read_trace(root)
    trace_counts: dict[str, int] = {}
    for event in trace:
        trace_counts[event.get("case_id", "")] = trace_counts.get(event.get("case_id", ""), 0) + 1
    cases = []
    for case_id in case_ids:
        if not CASE_ID.fullmatch(case_id):
            continue
        case = _read_json(root / "inputs" / f"{case_id}.json") or {}
        output = _read_json(root / "outputs" / f"{case_id}.json")
        request = case.get("customer_request") or {}
        cases.append(
            {
                "case_id": case_id,
                "opened_at": case.get("opened_at"),
                "topics": [c.get("topic") for c in request.get("claims") or []],
                "has_output": output is not None,
                "primary_issue": ((output or {}).get("assessment") or {}).get("primary_issue"),
                "case_status": ((output or {}).get("assessment") or {}).get("case_status"),
                "entity_status": ((output or {}).get("entity_resolution") or {}).get("status"),
                "trace_events": trace_counts.get(case_id, 0),
            }
        )
    return {
        "case_set_version": case_set.get("case_set_version"),
        "variant_id": case_set.get("variant_id"),
        "cases": cases,
        "trace_events": len(trace),
    }


def case_detail(root: Path, case_id: str) -> dict[str, Any] | None:
    case = _read_json(root / "inputs" / f"{case_id}.json")
    if case is None:
        return None
    return {
        "input": case,
        "output": _read_json(root / "outputs" / f"{case_id}.json"),
        "trace": [e for e in _read_trace(root) if e.get("case_id") == case_id],
    }


def make_handler(root: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, (UI_DIR / "index.html").read_bytes(), "text/html")
            elif path == "/api/summary":
                self._json(summary(root))
            elif path.startswith("/api/case/"):
                case_id = path.removeprefix("/api/case/")
                detail = case_detail(root, case_id) if CASE_ID.fullmatch(case_id) else None
                if detail is None:
                    self._json({"error": "case not found"}, HTTPStatus.NOT_FOUND)
                else:
                    self._json(detail)
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def _json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self._send(status, body, "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="L3B workflow viewer (read-only)")
    parser.add_argument("--root", default=".", help="repository root (default: .)")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(root))
    print(f"L3B viewer: http://127.0.0.1:{args.port}  (root={root}, Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
