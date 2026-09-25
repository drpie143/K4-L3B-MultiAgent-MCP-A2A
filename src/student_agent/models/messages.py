from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Type Aliases strictly aligned with contracts/schemas
EntityStatus = Literal["resolved", "ambiguous", "not_found"]

ShipmentVerdict = Literal[
    "on_time",
    "seller_delay",
    "logistics_delay",
    "lost",
    "returned",
    "conflicting",
    "insufficient_evidence",
]

PaymentVerdict = Literal[
    "reconciled",
    "capture_mismatch",
    "duplicate_capture",
    "refund_pending",
    "refund_failed",
    "refunded",
    "insufficient_evidence",
]

PartyType = Literal[
    "seller",
    "platform",
    "logistics_provider",
    "payment_provider",
    "customer",
    "unknown",
]


@dataclass
class RefundLine:
    """A single line item in financial_resolution.refund_lines."""

    reason_code: str
    amount_brl: float
    entity_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": str(self.reason_code),
            "amount_brl": round(float(self.amount_brl), 2),
            "entity_id": str(self.entity_id) if self.entity_id is not None else None,
        }


@dataclass
class FinancialResolution:
    """Strictly matches l3a-output-v2.schema.json#/$defs/financialResolution."""

    currency: Literal["BRL"] = "BRL"
    recommended_refund_brl: float = 0.0
    refund_lines: list[RefundLine] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": "BRL",
            "recommended_refund_brl": round(float(self.recommended_refund_brl), 2),
            "refund_lines": [line.to_dict() for line in self.refund_lines],
        }


@dataclass
class CandidateCause:
    """Specialist contribution to root_cause_analysis."""

    cause_code: str
    party_type: PartyType
    party_id: str | None = None
    rank: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause_code": str(self.cause_code),
            "party_type": str(self.party_type),
            "party_id": str(self.party_id) if self.party_id is not None else None,
            "rank": int(self.rank),
        }


@dataclass
class EntityResult:
    """Result of Person 1 (Entity Resolution + Customer Context)."""

    status: EntityStatus
    resolved_order_ids: list[str] = field(default_factory=list)
    rejected_candidates: list[str] = field(default_factory=list)
    confidence: float = 1.0
    customer_unique_id: str | None = None
    related_order_ids: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "resolved_order_ids": list(self.resolved_order_ids),
            "rejected_candidates": list(self.rejected_candidates),
            "confidence": float(self.confidence),
        }

    def to_customer_context_dict(self) -> dict[str, Any]:
        return {
            "customer_unique_id": self.customer_unique_id,
            "related_order_ids": list(self.related_order_ids),
        }


@dataclass
class ShipmentResult:
    """Result of Person 2 (Order/Product + Shipment Specialist)."""

    verdict: ShipmentVerdict
    late_seller_ids: list[str] = field(default_factory=list)
    timeline_complete: bool = True
    item_ids: list[str] = field(default_factory=list)
    seller_ids: list[str] = field(default_factory=list)
    shipment_ids: list[str] = field(default_factory=list)
    candidate_causes: list[CandidateCause] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "late_seller_ids": list(self.late_seller_ids),
            "timeline_complete": bool(self.timeline_complete),
        }


@dataclass
class PaymentResult:
    """Result of Person 3 (Payment / Refund / Financial Specialist).

    Strictly complies with payment_analysis and financial_resolution schemas.
    """

    verdict: PaymentVerdict
    captured_total_brl: float | None = None
    refunded_total_brl: float | None = None
    refundable_total_brl: float | None = None
    payment_references: list[str] = field(default_factory=list)
    financial_resolution: FinancialResolution = field(default_factory=FinancialResolution)
    candidate_causes: list[CandidateCause] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "captured_total_brl": (
                round(float(self.captured_total_brl), 2)
                if self.captured_total_brl is not None
                else None
            ),
            "refunded_total_brl": (
                round(float(self.refunded_total_brl), 2)
                if self.refunded_total_brl is not None
                else None
            ),
            "refundable_total_brl": (
                round(float(self.refundable_total_brl), 2)
                if self.refundable_total_brl is not None
                else None
            ),
        }
