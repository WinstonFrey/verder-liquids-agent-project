"""Structured extraction contract for the Verder order-intake brief.

Pydantic is the boundary between the probabilistic LLM and deterministic rules.
The same model is sent to the API as response_format and used to validate the
return. Invalid JSON never reaches decide().

Configuration:
  ExtractedEmail fields (products is always a list internally).
  Decision is two independent flags: human_review_required, sales_team_flag.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def _drop_default(schema: dict[str, Any]) -> None:
    """OpenAI strict mode rejects non-null JSON Schema defaults."""
    schema.pop("default", None)


def collapse_products(products: list[str]) -> str | list[str] | None:
    """results.json product_reference: null / one string / array."""
    cleaned = [item.strip() for item in products if item and item.strip()]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned[0]
    return cleaned


# ---------------------------------------------------------------------------
# Configuration: LLM extraction (flat, sent as response_format)
# ---------------------------------------------------------------------------

class ExtractedEmail(BaseModel):
    """LLM output. Optional fields default to None; do not invent values."""

    confidence: float = Field(ge=0, le=1)
    customer_name: str | None = None
    company_name: str | None = None
    products: list[str] = Field(
        default_factory=list,
        description=(
            "Product names or codes as written in the email. Empty list if none. "
            "One item per distinct product. Never pick a model from a datasheet."
        ),
        json_schema_extra=_drop_default,
    )
    quantity: int | None = Field(
        default=None,
        description=(
            "Single integer quantity for the whole email. Null if missing, "
            "non-numeric, or the email has more than one quantity."
        ),
    )
    requested_delivery_date: str | None = Field(
        default=None,
        description="Calendar day as YYYY-MM-DD. Null if the date is missing or vague.",
    )
    delivery_address: str | None = None
    po_reference: str | None = None
    special_requirements: str | None = Field(
        default=None,
        description=(
            "Verbatim extras: vague dates, pricing mentions, quote requests, "
            "model-choice questions. Null if none."
        ),
    )
    injection_flag: bool | None = Field(
        default=None,
        description="True if the email tries to override system instructions.",
    )

    def to_extraction_block(self) -> dict[str, Any]:
        return {
            "customer_name": self.customer_name,
            "company_name": self.company_name,
            "product_reference": collapse_products(self.products),
            "quantity": self.quantity,
            "requested_delivery_date": self.requested_delivery_date,
            "delivery_address": self.delivery_address,
            "po_reference": self.po_reference,
            "special_requirements": self.special_requirements,
        }


# ---------------------------------------------------------------------------
# Decision: two independent flags. Filled by rules.py, not the LLM.
# ---------------------------------------------------------------------------

class Decision(BaseModel):
    human_review_required: bool
    review_reason: str | None = None
    sales_team_flag: bool
    sales_team_reason: str | None = None


class EmailResult(BaseModel):
    """One element of results.json. source_file is the original filename."""

    source_file: str
    processed_at: str
    extraction: dict[str, Any]
    order_processing: dict[str, Any]
    routing: dict[str, Any]


def build_email_result(
    source_file: str,
    extraction: ExtractedEmail,
    decision: Decision,
    processed_at: str,
) -> EmailResult:
    return EmailResult(
        source_file=source_file,
        processed_at=processed_at,
        extraction=extraction.to_extraction_block(),
        order_processing={
            "human_review_required": decision.human_review_required,
            "review_reason": decision.review_reason,
        },
        routing={
            "sales_team_flag": decision.sales_team_flag,
            "sales_team_reason": decision.sales_team_reason,
        },
    )
