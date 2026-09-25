"""Case-scoped MCP response cache.

Keys are ``(case_id, tool_name, arguments)``. A hit is never reused for another case.
"""

from __future__ import annotations

from typing import Any


def _key(
    case_id: str, tool_name: str, arguments: dict[str, str]
) -> tuple[str, str, tuple[tuple[str, str], ...]]:
    items = tuple(sorted((str(key), str(value)) for key, value in arguments.items()))
    return case_id, tool_name, items


class CaseCache:
    """Remember both successful payloads and confirmed misses for one process."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str, tuple[tuple[str, str], ...]], Any] = {}
        self._seen: set[tuple[str, str, tuple[tuple[str, str], ...]]] = set()

    def get(self, case_id: str, tool_name: str, arguments: dict[str, str]) -> tuple[bool, Any]:
        key = _key(case_id, tool_name, arguments)
        if key not in self._seen:
            return False, None
        return True, self._values.get(key)

    def put(self, case_id: str, tool_name: str, arguments: dict[str, str], value: Any) -> None:
        key = _key(case_id, tool_name, arguments)
        self._seen.add(key)
        self._values[key] = value
