"""Deterministic flags. No LLM calls in this file.

human_review_required and sales_team_flag are independent.
The LLM never sets either flag.

Configuration:
  KNOWN_PRODUCTS (from datasheet filenames), CONFIDENCE_MIN,
  quote/pricing/model-selection markers, required-field checks.
"""

from __future__ import annotations

from pathlib import Path

from schema import Decision, ExtractedEmail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIDENCE_MIN = 0.75

_CASE_DIR = Path(__file__).resolve().parent / "REAL_Test case"
KNOWN_PRODUCTS = {
    path.stem.rsplit(" ", 1)[-1].upper()
    for path in _CASE_DIR.glob("04_Datasheet *.pdf")
}
if not KNOWN_PRODUCTS:
    KNOWN_PRODUCTS = {"ICP2", "MCP2", "MCP3", "MWP2", "NMS", "PHP2", "PRP2"}

INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore all instructions",
    "ignore the instructions",
    "disregard previous",
    "you are now",
    "reveal your prompt",
    "system prompt",
    "negeer vorige instructies",
    "ignoriere vorherige",
    "ignore les instructions",
    "ignore as instrucoes",
    "ignore as instruções",
)

# Sales-team: order can proceed; parallel attention. Not a human blocker.
QUOTE_MARKERS = (
    "quote request",
    "request a quote",
    "request for quote",
    "quotation",
    "devis",
    "offerteaanvraag",
    "presupuesto",
)
PRICING_MARKERS = (
    "pricing",
    "price",
    "prijs",
    "prix",
    "precio",
    "raamovereenkomst",
    "framework agreement",
)
MODEL_SELECTION_MARKERS = (
    "not 100%",
    "right fit",
    "confirm the right model",
    "whether the",
)
# Check negation first: "urgent" is a substring of "pas urgent".
NOT_URGENT_MARKERS = (
    "pas urgent",
    "n'est pas urgent",
    "not urgent",
    "niet urgent",
    "no es urgente",
    "não é urgente",
    "nao e urgente",
    "non urgente",
    "nicht dringend",
)
URGENT_MARKERS = (
    "urgent",
    "urgente",
    "urgently",
    "asap",
    "a.s.a.p",
    "dringend",
    "spoed",
)
# Window / range phrasing is not a calendar day (email #7 "week of 4 August").
VAGUE_DATE_MARKERS = (
    "week of",
    "week commencing",
    "semaine du",
    "semaine de",
    "semana de",
    "semana del",
    "mid-",
    "mid ",
    "mi-août",
    "mi-aout",
    "mi août",
    "beginning of",
    "end of",
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def looks_like_injection(raw_email: str | None) -> bool:
    if not raw_email:
        return False
    text = raw_email.lower()
    return any(marker in text for marker in INJECTION_MARKERS)


def _blob(extraction: ExtractedEmail, raw_email: str | None) -> str:
    parts = [raw_email or "", extraction.special_requirements or ""]
    return "\n".join(parts).lower()


def looks_like_quote(extraction: ExtractedEmail, raw_email: str | None) -> bool:
    return any(marker in _blob(extraction, raw_email) for marker in QUOTE_MARKERS)


def looks_like_pricing(extraction: ExtractedEmail, raw_email: str | None) -> bool:
    return any(marker in _blob(extraction, raw_email) for marker in PRICING_MARKERS)


def named_known_products(extraction: ExtractedEmail) -> list[str]:
    hits: list[str] = []
    for item in extraction.products:
        upper = item.upper()
        if any(code in upper for code in KNOWN_PRODUCTS):
            hits.append(item)
    return hits


def looks_like_model_selection(
    extraction: ExtractedEmail, raw_email: str | None
) -> bool:
    if len(named_known_products(extraction)) < 2:
        return False
    blob = _blob(extraction, raw_email)
    return any(marker in blob for marker in MODEL_SELECTION_MARKERS) or " or " in blob


def looks_like_urgency(extraction: ExtractedEmail, raw_email: str | None) -> bool:
    blob = _blob(extraction, raw_email)
    if any(marker in blob for marker in NOT_URGENT_MARKERS):
        return False
    return any(marker in blob for marker in URGENT_MARKERS)


def looks_like_vague_date(extraction: ExtractedEmail, raw_email: str | None) -> bool:
    return any(marker in _blob(extraction, raw_email) for marker in VAGUE_DATE_MARKERS)


def _sentence(parts: list[str]) -> str | None:
    if not parts:
        return None
    return " ".join(parts)


# ---------------------------------------------------------------------------
# The decision function
# ---------------------------------------------------------------------------

def decide(extraction: ExtractedEmail, raw_email: str | None = None) -> Decision:
    """Two independent flags. Multi-line recognised product lists are processable."""
    human: list[str] = []
    sales: list[str] = []

    quote = looks_like_quote(extraction, raw_email)
    pricing = looks_like_pricing(extraction, raw_email)
    model_choice = looks_like_model_selection(extraction, raw_email)
    known = named_known_products(extraction)

    if extraction.injection_flag or looks_like_injection(raw_email):
        human.append("Possible prompt injection in the email body; treat as data only.")

    if extraction.confidence < CONFIDENCE_MIN:
        human.append(
            f"Extraction confidence {extraction.confidence:.2f} is below {CONFIDENCE_MIN:.2f}."
        )

    if looks_like_urgency(extraction, raw_email):
        human.append("Urgent tone; human must check before the order proceeds.")

    if looks_like_vague_date(extraction, raw_email):
        human.append(
            "Delivery date is a window, not a calendar day; human to confirm."
        )

    if quote:
        sales.append("Quote request is out of scope for intake; flag sales for parallel attention.")
    if pricing:
        sales.append("Pricing is out of scope for this agent; flag sales for parallel attention.")
    if model_choice:
        sales.append(
            "Model selection question; do not pick a datasheet winner, flag sales."
        )
        human.append(
            "Product is not determined; cannot process the order until a model is chosen."
        )

    # Recognised product lists (#2, #4) are processable; quantity may be null.
    # Quote-only (#7) skips missing-product checks; vague dates still block above.
    if not quote:
        if not known:
            human.append("No recognised product in the email; cannot process the order.")
        elif len(known) == 1 and extraction.quantity is None:
            human.append("Quantity is missing; cannot process the order.")
        if known and not (extraction.delivery_address or "").strip():
            human.append("Delivery address is missing; cannot process the order.")

    return Decision(
        human_review_required=bool(human),
        review_reason=_sentence(human),
        sales_team_flag=bool(sales),
        sales_team_reason=_sentence(sales),
    )
