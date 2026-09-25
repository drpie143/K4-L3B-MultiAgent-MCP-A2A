"""Persistent, case-scoped store of MCP evidence envelopes.

Records each ``(case_id, tool, arguments)`` envelope once and replays it from disk, so the
workflow logic can be re-run offline on real data without new MCP calls.

Layout: ``<root>/<case_id>/<tool>__<sha1(arguments)>.json``. Keys always include the
case_id, so evidence is never shared across cases. Deterministic tool errors are stored
too, so a wrong candidate is not looked up twice; transport errors are never stored.

Off by default. The scorer ties every evidence_ref to the MCP run (connection) that issued
it, so a submission must come from one uninterrupted live run; replayed refs from an earlier
run are cross-scope. Enable only for offline analysis: ``DAY09_EVIDENCE_STORE=.mcp_store``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from ..cases import CASE_ID_PATTERN

DEFAULT_ROOT = ".mcp_store"
TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def store_root() -> Path | None:
    value = os.getenv("DAY09_EVIDENCE_STORE", "off").strip()
    if value.lower() in {"", "off", "0", "false", "no"}:
        return None
    return Path(value).resolve()


class StoredGateway:
    """Gateway wrapper that replays stored envelopes and records new ones."""

    def __init__(self, gateway: Any, root: Path) -> None:
        self._gateway = gateway
        self._root = root
        self._tools: list[str] | None = None
        self.live_calls = 0
        self.replayed_calls = 0

    async def list_tools(self) -> list[str]:
        if self._tools is None:
            self._tools = list(await self._gateway.list_tools())
        return list(self._tools)

    def _path(self, tool_name: str, case_id: str, arguments: dict[str, str]) -> Path:
        if not CASE_ID_PATTERN.fullmatch(case_id) or not TOOL_NAME.fullmatch(tool_name):
            raise ValueError("invalid case_id or tool name for evidence store")
        canonical = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
        return self._root / case_id / f"{tool_name}__{digest}.json"

    async def call(self, tool_name: str, *, case_id: str, **arguments: str) -> dict[str, Any]:
        path = self._path(tool_name, case_id, arguments)
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            self.replayed_calls += 1
            if "error" in record:
                raise RuntimeError(record["error"])
            return record["evidence"]
        self.live_calls += 1
        try:
            evidence = await self._gateway.call(tool_name, case_id=case_id, **arguments)
        except RuntimeError as exc:  # deterministic tool error (e.g. unknown order)
            self._write(path, {"tool": tool_name, "arguments": arguments, "error": str(exc)})
            raise
        self._write(path, {"tool": tool_name, "arguments": arguments, "evidence": evidence})
        return evidence

    @staticmethod
    def _write(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)
