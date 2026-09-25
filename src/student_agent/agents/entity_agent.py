"""Entity resolution + customer context agent.

Resolves which order(s) a case is about using only MCP evidence scoped to the case:

1. Read claimed order IDs, candidate order IDs and customer hints from the raw case.
2. If a ``customer_unique_id`` hint exists, fetch the customer history once and reject
   candidates that are not in it (saves one ``get_order`` call per rejected candidate).
3. Call ``get_order`` for the remaining candidates, reject those that do not exist or
   contradict a hard identifier, then score the rest against soft hints.
4. Resolve only when evidence singles out an order; otherwise report ``ambiguous`` or
   ``not_found``. The first candidate is never picked by default.

Trace: ``task_assigned`` (coordinator -> entity-agent), one ``tool_result_consumed`` per
MCP evidence, and a final ``handoff`` back to the coordinator. No reasoning is traced.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ..cases import CASE_ID_PATTERN
from ..models.messages import EntityResult

ACTOR = "entity-agent"
COORDINATOR = "coordinator"
TOOL_ORDER = "get_order"
TOOL_HISTORY = "get_customer_history"

MAX_ORDER_LOOKUPS = 8
MAX_ATTEMPTS = 2  # one retry for transport failures only; tool errors are not retried
MAX_IDS = 20  # idSet maxItems in the output schema
MAX_TRACE_REFS = 20  # evidence_refs maxItems in the trace schema

AMBIGUOUS_CONFIDENCE_CAP = 0.5
NOT_FOUND_CONFIDENCE = 0.1

CLAIMED_ORDER_KEYS = (
    "order_id",
    "order_ids",
    "reported_order_id",
    "claimed_order_id",
    "claimed_order_ids",
)
CANDIDATE_KEYS = (
    "candidate_order_ids",
    "candidate_orders",
    "order_candidates",
    "candidates",
    "possible_order_ids",
    "possible_orders",
)
CANDIDATE_ID_KEYS = ("order_id", "candidate_order_id", "id")
FREE_TEXT_KEYS = ("message", "complaint", "customer_message", "description", "text", "body")
ORDER_ID_IN_TEXT = re.compile(r"\b[0-9a-f]{32}\b")
DATE_PREFIX = re.compile(r"^\d{4}-\d{2}(-\d{2})?")

# A mismatch on a hard hint rejects the candidate outright.
HARD_HINTS: dict[str, tuple[str, ...]] = {
    "customer_unique_id": ("customer_unique_id",),
    "customer_id": ("customer_id",),
}
# Soft hints only add to or subtract from a candidate's score.
SOFT_HINTS: dict[str, tuple[str, ...]] = {
    "purchase_date": (
        "order_purchase_timestamp",
        "purchase_date",
        "purchase_timestamp",
        "order_date",
        "purchased_at",
        "ordered_at",
    ),
    "customer_city": ("customer_city",),
    "customer_state": ("customer_state",),
    "customer_zip_code_prefix": ("customer_zip_code_prefix", "zip_code_prefix"),
}


@dataclass
class EntityAgentResult(EntityResult):
    """EntityResult plus optional extras; consumers may rely on EntityResult fields only."""

    # raw get_order evidence of resolved orders, so other agents need not re-call it.
    order_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    # refs that only justify rejecting a candidate (kept apart to protect evidence precision).
    rejection_evidence_refs: list[str] = field(default_factory=list)
    # candidate order_id -> decision code, e.g. RESOLVED, NOT_FOUND, CUSTOMER_MISMATCH.
    candidate_decisions: dict[str, str] = field(default_factory=dict)


@dataclass(eq=False)
class _Candidate:
    order_id: str
    claimed: bool = False
    evidence: dict[str, Any] | None = None
    decision: str | None = None
    matches: int = 0
    mismatches: int = 0

    @property
    def score(self) -> int:
        return self.matches - self.mismatches


class _CaseSession:
    """MCP access bound to one case: fixed case_id, per-case cache, bounded retry."""

    def __init__(self, case_id: str, gateway: Any, trace: Any, tools: set[str] | None) -> None:
        self.case_id = case_id
        self.gateway = gateway
        self.trace = trace
        self.tools = tools
        self.calls = 0
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any] | None] = {}

    def allows(self, tool_name: str) -> bool:
        return self.tools is None or tool_name in self.tools

    async def call(self, tool_name: str, **arguments: str) -> dict[str, Any] | None:
        key = (tool_name, tuple(sorted(arguments.items())))
        if key in self._cache:
            return self._cache[key]
        evidence: dict[str, Any] | None = None
        if self.allows(tool_name):
            for attempt in range(1, MAX_ATTEMPTS + 1):
                self.calls += 1
                try:
                    evidence = await self.gateway.call(tool_name, case_id=self.case_id, **arguments)
                    break
                except Exception:
                    if attempt == MAX_ATTEMPTS:
                        break
                    import asyncio
                    await asyncio.sleep(2 * attempt)
        self._cache[key] = evidence
        return evidence

    def consumed(self, tool_name: str, evidence: dict[str, Any]) -> None:
        self.trace.emit(
            case_id=self.case_id,
            event_type="tool_result_consumed",
            actor=ACTOR,
            tool_name=tool_name,
            evidence_refs=[evidence["evidence_ref"]],
            attributes={"domain": evidence.get("domain")},
        )


async def resolve_entity(
    case: dict[str, Any],
    gateway: Any,
    trace: Any,
    *,
    available_tools: set[str] | list[str] | None = None,
    fetch_history: bool = True,
) -> EntityAgentResult:
    """Resolve the case's order(s) and customer context from MCP evidence.

    ``available_tools`` should come from tool discovery (``gateway.list_tools()``); tools
    not listed are never called. ``fetch_history`` controls the customer history lookup
    used for ``customer_context.related_order_ids``.
    """
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(f"invalid case_id: {case_id!r}")
    trace.emit(
        case_id=case_id,
        event_type="task_assigned",
        actor=COORDINATOR,
        target=ACTOR,
        decision_code="RESOLVE_ENTITY",
    )
    tools = set(available_tools) if available_tools is not None else None
    session = _CaseSession(case_id, gateway, trace, tools)

    claimed, candidate_ids, text_ids = extract_order_ids(case)
    hints = extract_hints(case)
    hint_customer = hints.get("customer_unique_id")

    pool: list[_Candidate] = [
        _Candidate(order_id, claimed=order_id in claimed)
        for order_id in _unique([*claimed, *candidate_ids])
    ]
    from_text = not pool
    if from_text:
        pool = [_Candidate(order_id) for order_id in text_ids]

    # Customer history first: one call can reject several candidates at once.
    history_evidence: dict[str, Any] | None = None
    history_ids: list[str] = []
    if hint_customer and session.allows(TOOL_HISTORY) and (fetch_history or len(pool) != 1):
        history_evidence = await _fetch_history(session, hint_customer)
        if history_evidence is not None:
            history_ids = collect_order_ids(history_evidence.get("data"))

    from_history = not pool and bool(history_ids)
    if from_history:
        pool = [_Candidate(order_id) for order_id in history_ids[:MAX_ORDER_LOOKUPS]]

    if history_ids and not from_history:
        known = set(history_ids)
        for candidate in pool:
            if candidate.order_id not in known:
                candidate.decision = "NOT_IN_CUSTOMER_HISTORY"

    pending = [candidate for candidate in pool if candidate.decision is None]
    to_lookup, unassessed = pending[:MAX_ORDER_LOOKUPS], pending[MAX_ORDER_LOOKUPS:]
    responses = await asyncio.gather(
        *(session.call(TOOL_ORDER, order_id=c.order_id) for c in to_lookup)
    )
    for candidate, evidence in zip(to_lookup, responses, strict=True):
        if evidence is not None:
            session.consumed(TOOL_ORDER, evidence)
        _assess(candidate, evidence, hints, bonus=bool(candidate_ids) and candidate.claimed)
    if from_text:
        # IDs scraped from free text are not declared candidates; drop the ones that fail.
        pool = [c for c in pool if c.decision != "NOT_FOUND"]

    survivors = [c for c in pool if c.decision is None and c not in unassessed]
    resolved = _select(survivors, unassessed, multi_claim=len(claimed) > 1 and not candidate_ids)
    for candidate in survivors:
        if candidate in resolved:
            candidate.decision = "RESOLVED"
        elif resolved:
            candidate.decision = "LOWER_MATCH_SCORE"
    if not resolved and survivors:
        top = max(c.score for c in survivors)
        for candidate in survivors:
            if candidate.score < top:
                candidate.decision = "LOWER_MATCH_SCORE"

    status = "resolved" if resolved else ("ambiguous" if survivors or unassessed else "not_found")
    resolved_ids = [c.order_id for c in resolved]
    customer_unique_id = _customer_identity(resolved, hint_customer, history_evidence, history_ids)

    if (
        resolved
        and customer_unique_id
        and history_evidence is None
        and fetch_history
        and session.allows(TOOL_HISTORY)
    ):
        history_evidence = await _fetch_history(session, customer_unique_id)
        if history_evidence is not None:
            history_ids = collect_order_ids(history_evidence.get("data"))

    rejected = [c for c in pool if c.decision not in (None, "RESOLVED")]
    open_ = [c for c in pool if c.decision is None]
    refs, rejection_refs = _partition_refs(status, pool, history_evidence)

    result = EntityAgentResult(
        status=status,
        resolved_order_ids=resolved_ids[:MAX_IDS],
        rejected_candidates=[c.order_id for c in rejected][:MAX_IDS],
        confidence=_confidence(
            status, resolved, open_, claimed_only=bool(claimed) and not candidate_ids
        ),
        customer_unique_id=customer_unique_id,
        related_order_ids=[i for i in history_ids if i not in resolved_ids][:MAX_IDS],
        evidence_refs=refs,
        order_evidence={c.order_id: c.evidence for c in resolved if c.evidence is not None},
        rejection_evidence_refs=rejection_refs,
        candidate_decisions={c.order_id: c.decision or "UNRESOLVED" for c in pool},
    )
    trace.emit(
        case_id=case_id,
        event_type="handoff",
        actor=ACTOR,
        target=COORDINATOR,
        decision_code=f"ENTITY_{status.upper()}",
        evidence_refs=result.evidence_refs[:MAX_TRACE_REFS] or None,
        attributes={
            "status": status,
            "candidate_count": len(pool),
            "resolved_count": len(result.resolved_order_ids),
            "rejected_count": len(result.rejected_candidates),
            "confidence": result.confidence,
            "customer_identified": customer_unique_id is not None,
            "mcp_calls": session.calls,
        },
    )
    return result


def entity_output_fields(result: EntityResult) -> dict[str, Any]:
    """Map an EntityResult onto the output schema sections this agent owns."""
    return {
        "entity_resolution": result.to_dict(),
        "customer_context": result.to_customer_context_dict(),
        "order_ids": list(result.resolved_order_ids),
        "evidence_refs": list(result.evidence_refs),
    }


# --- case parsing -------------------------------------------------------------------


def extract_order_ids(case: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return (claimed order IDs, candidate order IDs, IDs found in free text)."""
    claimed: list[str] = []
    candidates: list[str] = []
    text_ids: list[str] = []
    for key, value in _scan(case):
        if key in CLAIMED_ORDER_KEYS:
            claimed.extend(_ids_from(value))
        elif key in CANDIDATE_KEYS:
            candidates.extend(_ids_from(value))
        elif key in FREE_TEXT_KEYS and isinstance(value, str):
            text_ids.extend(ORDER_ID_IN_TEXT.findall(value))
    return _unique(claimed), _unique(candidates), _unique(text_ids)


def extract_hints(case: dict[str, Any]) -> dict[str, str]:
    """Customer-side identifiers and attributes usable to match candidate orders."""
    hints: dict[str, str] = {}
    aliases = {**HARD_HINTS, **SOFT_HINTS}
    for key, value in _scan(case):
        for name, names in aliases.items():
            if key in names and name not in hints and _scalar(value):
                hints[name] = str(value).strip()
    return hints


def collect_order_ids(data: Any) -> list[str]:
    """All order IDs listed anywhere in an evidence payload (e.g. customer history)."""
    found: list[str] = []
    stack = [data]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("order_id", "order_ids"):
                    found.extend(_ids_from(value))
                elif isinstance(value, dict | list):
                    stack.append(value)
        elif isinstance(node, list):
            stack.extend(node)
    return _unique(found)


def _scan(node: Any, depth: int = 0, max_depth: int = 2) -> Iterator[tuple[str, Any]]:
    """Yield key/value pairs of nested dicts, without descending into lists or candidates."""
    if not isinstance(node, dict) or depth > max_depth:
        return
    for key, value in node.items():
        yield key, value
        if key not in CANDIDATE_KEYS and isinstance(value, dict):
            yield from _scan(value, depth + 1, max_depth)


def _ids_from(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if 0 < len(text) <= 128 else []
    if isinstance(value, list):
        return [order_id for item in value for order_id in _ids_from(item)]
    if isinstance(value, dict):
        for key in (*CANDIDATE_ID_KEYS, "order_ids"):
            if key in value:
                return _ids_from(value[key])
    return []


def _scalar(value: Any) -> bool:
    return (
        isinstance(value, str | int | float)
        and not isinstance(value, bool)
        and bool(str(value).strip())
    )


def _unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# --- evidence assessment ------------------------------------------------------------


async def _fetch_history(session: _CaseSession, customer_unique_id: str) -> dict[str, Any] | None:
    evidence = await session.call(TOOL_HISTORY, customer_unique_id=customer_unique_id)
    if evidence is not None:
        session.consumed(TOOL_HISTORY, evidence)
    return evidence


def find_order_record(data: Any, order_id: str) -> dict[str, Any] | None:
    """The row describing ``order_id`` in a get_order payload, or None if absent."""
    if data in (None, {}, []) or (isinstance(data, dict) and data.get("found") is False):
        return None
    saw_order_id = False
    stack = [data]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            if "order_id" in node:
                saw_order_id = True
                if str(node["order_id"]) == order_id:
                    return node
            stack.extend(v for v in node.values() if isinstance(v, dict | list))
        elif isinstance(node, list):
            stack.extend(node)
    # Payload without any order_id field: the scoped tool answered for this order.
    if not saw_order_id and isinstance(data, dict):
        return data
    return None


def _field(data: Any, record: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for source in (record, data):
        for key, value in _scan(source):
            if key in names and _scalar(value):
                return str(value).strip()
    return None


def _same(name: str, expected: str, actual: str) -> bool:
    if name == "purchase_date" and DATE_PREFIX.match(expected) and DATE_PREFIX.match(actual):
        width = len(DATE_PREFIX.match(expected).group(0))
        return actual[:width] == expected[:width]
    return expected.casefold() == actual.casefold()


def _assess(
    candidate: _Candidate, evidence: dict[str, Any] | None, hints: dict[str, str], *, bonus: bool
) -> None:
    candidate.evidence = evidence
    data = evidence.get("data") if evidence is not None else None
    record = find_order_record(data, candidate.order_id)
    if record is None:
        candidate.decision = "NOT_FOUND"
        return
    for name, names in HARD_HINTS.items():
        actual = _field(data, record, names)
        if name in hints and actual is not None and not _same(name, hints[name], actual):
            candidate.decision = "CUSTOMER_MISMATCH"
            return
        if name in hints and actual is not None:
            candidate.matches += 1
    for name, names in SOFT_HINTS.items():
        actual = _field(data, record, names)
        if name not in hints or actual is None:
            continue
        if _same(name, hints[name], actual):
            candidate.matches += 1
        else:
            candidate.mismatches += 1
    if bonus:
        candidate.matches += 1  # the customer named this order explicitly


def _select(
    survivors: list[_Candidate], unassessed: list[_Candidate], *, multi_claim: bool
) -> list[_Candidate]:
    """Pick resolved candidates; empty means the evidence does not single one out."""
    if not survivors or unassessed:
        return []
    if multi_claim:
        return [c for c in survivors if c.claimed]
    if len(survivors) == 1:
        only = survivors[0]
        return [only] if only.claimed or only.mismatches <= only.matches else []
    ranked = sorted(survivors, key=lambda c: c.score, reverse=True)
    top, runner_up = ranked[0], ranked[1]
    if top.score > runner_up.score and top.matches >= 1 and top.mismatches == 0:
        return [top]
    return []


def _customer_identity(
    resolved: list[_Candidate],
    hint: str | None,
    history_evidence: dict[str, Any] | None,
    history_ids: list[str],
) -> str | None:
    from_orders = {
        value
        for c in resolved
        if c.evidence is not None
        and (record := find_order_record(c.evidence.get("data"), c.order_id)) is not None
        and (value := _field(c.evidence.get("data"), record, ("customer_unique_id",)))
    }
    if len(from_orders) == 1:
        return from_orders.pop()
    if len(from_orders) > 1:
        return None  # resolved orders disagree on the customer
    if hint and history_evidence is not None and history_ids:
        known = set(history_ids)
        if not resolved or all(c.order_id in known for c in resolved):
            return hint
    return None


def _partition_refs(
    status: str, pool: list[_Candidate], history: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    history_refs = [history["evidence_ref"]] if history is not None else []
    order_refs = {c.order_id: c.evidence["evidence_ref"] for c in pool if c.evidence is not None}
    if status == "resolved":
        supporting = [order_refs[c.order_id] for c in pool if c.decision == "RESOLVED"]
        refs = _unique([*supporting, *history_refs])
        rejection = [r for r in _unique(list(order_refs.values())) if r not in refs]
        return refs, rejection
    # Unresolved: every consulted ref is needed to justify the status.
    return _unique([*order_refs.values(), *history_refs]), []


def _confidence(
    status: str, resolved: list[_Candidate], open_: list[_Candidate], *, claimed_only: bool
) -> float:
    if status == "resolved":
        base = 0.9 if claimed_only else 0.75
        scores = [base + 0.05 * min(c.matches, 3) - 0.15 * c.mismatches for c in resolved]
        return round(min(0.97, max(0.5, min(scores))), 2)
    if status == "ambiguous":
        return round(min(AMBIGUOUS_CONFIDENCE_CAP, 1 / max(len(open_), 1)), 2)
    return NOT_FOUND_CONFIDENCE
