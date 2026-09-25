"""Pure helpers for reading MCP payloads and shaping schema-safe ids."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

EVIDENCE_REF_RE = re.compile(r"^ev_[A-Za-z0-9_-]{20,96}$")
CAUSE_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,79}$")

CARRIER_KEYS = (
    "order_delivered_carrier_date",
    "delivered_carrier_date",
    "seller_handoff_at",
    "handoff_at",
)
CUSTOMER_DELIVERY_KEYS = (
    "order_delivered_customer_date",
    "delivered_customer_date",
    "actual_delivery_date",
    "delivered_at",
)
ESTIMATED_DELIVERY_KEYS = (
    "order_estimated_delivery_date",
    "estimated_delivery_date",
    "expected_delivery_date",
)
SHIPPING_LIMIT_KEYS = (
    "shipping_limit_date",
    "seller_shipping_limit",
    "handoff_limit",
)


def walk_dicts(data: Any) -> list[dict[str, Any]]:
    """Dict nodes in document order, outer containers before their children."""
    found: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            found.append(node)
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(data)
    return found


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def round_money(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, float(value)), 2)


def parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt, size in (("%Y-%m-%d %H:%M:%S", 19), ("%Y-%m-%d", 10)):
        try:
            return datetime.strptime(text[:size], fmt)
        except ValueError:
            continue
    return None


def as_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def first_text(data: Any, names: tuple[str, ...]) -> str | None:
    for node in walk_dicts(data):
        for name in names:
            value = node.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, int | float) and not isinstance(value, bool):
                return str(value)
    return None


def first_datetime(data: Any, names: tuple[str, ...]) -> datetime | None:
    for node in walk_dicts(data):
        for name in names:
            if name in node:
                parsed = parse_datetime(node.get(name))
                if parsed is not None:
                    return parsed
    return None


def unique_ids(values: list[Any], limit: int = 20) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or len(text) > 128 or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def valid_evidence_refs(values: list[Any], limit: int = 30) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not EVIDENCE_REF_RE.fullmatch(value) or value in seen:
            continue
        seen.add(value)
        result.append(value)
        if len(result) >= limit:
            break
    return result


def evidence_ref_of(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    ref = payload.get("evidence_ref")
    if isinstance(ref, str) and EVIDENCE_REF_RE.fullmatch(ref):
        return ref
    return None
