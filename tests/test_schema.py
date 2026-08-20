"""Schema and prompt contract. No network.

    python tests/test_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from prompts import assert_prompt_matches_schema, build_system_prompt, wrap_email
from schema import (
    Decision,
    ExtractedEmail,
    build_email_result,
    collapse_products,
)


EMAIL_1 = ROOT / "REAL_Test case" / "02_Email #1.txt"


def main() -> None:
    assert_prompt_matches_schema()
    system = build_system_prompt()
    assert "untrusted DATA" in system
    assert "Supported languages" not in system
    assert "datasheet" in system.lower()
    raw = EMAIL_1.read_text(encoding="utf-8")
    wrapped = wrap_email(raw)
    assert "UNTRUSTED EMAIL START" in wrapped
    assert "PO-2026-0441" in wrapped

    extracted = ExtractedEmail.model_validate(
        {
            "confidence": 0.92,
            "customer_name": "Jan Van den Berghe",
            "company_name": "Van Belle Industries NV",
            "products": ["Packo ICP2"],
            "quantity": 2,
            "requested_delivery_date": "2026-07-28",
            "delivery_address": "Industrieweg 14, 9000 Ghent, Belgium",
            "po_reference": "PO-2026-0441",
        }
    )
    block = extracted.to_extraction_block()
    assert block["product_reference"] == "Packo ICP2"
    assert block["quantity"] == 2
    assert block["special_requirements"] is None

    assert collapse_products([]) is None
    assert collapse_products(["Packo NMS", "Packo MWP2"]) == [
        "Packo NMS",
        "Packo MWP2",
    ]

    try:
        ExtractedEmail.model_validate({"confidence": 1.5, "products": ["ICP2"]})
        raise AssertionError("confidence 1.5 should fail")
    except ValidationError:
        pass

    # Email 1 expected flags live in rules.py next. Shape only, here.
    result = build_email_result(
        source_file=EMAIL_1.name,
        extraction=extracted,
        decision=Decision(
            human_review_required=False,
            review_reason=None,
            sales_team_flag=False,
            sales_team_reason=None,
        ),
        processed_at="2026-08-19T10:00:00+00:00",
    )
    dumped = result.model_dump()
    assert dumped["source_file"] == "02_Email #1.txt"
    assert dumped["order_processing"]["human_review_required"] is False
    assert dumped["routing"]["sales_team_flag"] is False

    print("test_schema: all checks passed")
    print(f"  source_file         -> {dumped['source_file']}")
    print(f"  product_reference   -> {block['product_reference']!r}")
    print(f"  quantity            -> {block['quantity']}")
    print(f"  date                -> {block['requested_delivery_date']}")


if __name__ == "__main__":
    main()
