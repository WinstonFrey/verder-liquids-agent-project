"""Batch runner: emails in, results.json out.

    python harness.py
    python harness.py --dir "REAL_Test case"
    python harness.py --out results.json

Pipeline per email: extract_email() -> decide(extraction, raw_email=email).
source_file is the original filename. Do not rename the brief's files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from llm import (
    ExtractionResult,
    Provider,
    default_providers,
    estimate_cost_usd,
    extract_email,
)
from rules import decide
from schema import Decision, ExtractedEmail, build_email_result, collapse_products

EMAIL_DIR = Path("REAL_Test case")
OUTPUT_PATH = Path("results.json")


@dataclass
class CaseResult:
    name: str
    email: str
    llm: ExtractionResult
    decision: Decision


def _failed_decision(error: str) -> Decision:
    return Decision(
        human_review_required=True,
        review_reason=f"Extraction failed: {error}",
        sales_team_flag=False,
        sales_team_reason=None,
    )


def process_email(
    name: str, email: str, providers: list[Provider] | None = None
) -> CaseResult:
    result = extract_email(email, providers=providers)
    if result.ok and result.extraction is not None:
        decision = decide(result.extraction, raw_email=email)
    else:
        decision = _failed_decision(result.error or "unknown extraction error")
    return CaseResult(name=name, email=email, llm=result, decision=decision)


def load_emails(directory: Path = EMAIL_DIR) -> list[tuple[str, str]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"email directory not found: {directory}")
    files = sorted(directory.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no .txt emails in {directory}")
    return [(path.name, path.read_text(encoding="utf-8")) for path in files]


def run(
    directory: Path = EMAIL_DIR,
    limit: int | None = None,
    providers: list[Provider] | None = None,
) -> list[CaseResult]:
    emails = load_emails(directory)
    if limit:
        emails = emails[:limit]
    cases: list[CaseResult] = []
    for index, (name, text) in enumerate(emails, start=1):
        print(f"[{index}/{len(emails)}] {name} ...", flush=True)
        try:
            cases.append(process_email(name, text, providers=providers))
        except Exception as exc:
            failed = ExtractionResult(
                ok=False,
                extraction=None,
                raw="",
                provider="none",
                model="none",
                mode="none",
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=0,
                attempts=[],
            )
            cases.append(
                CaseResult(
                    name=name,
                    email=text,
                    llm=failed,
                    decision=_failed_decision(f"{type(exc).__name__}: {exc}"),
                )
            )
    return cases


def write_results(
    cases: list[CaseResult], path: Path, processed_at: str
) -> None:
    payload = []
    for case in cases:
        extraction = case.llm.extraction or ExtractedEmail(confidence=0.0)
        payload.append(
            build_email_result(
                source_file=case.name,
                extraction=extraction,
                decision=case.decision,
                processed_at=processed_at,
            ).model_dump()
        )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_COLS = (
    ("case", 22),
    ("products", 22),
    ("qty", 4),
    ("date", 10),
    ("HR", 3),
    ("ST", 3),
    ("conf", 4),
    ("via", 16),
)


def _fit(text: object, width: int) -> str:
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat.ljust(width)
    return flat[: width - 2] + ".."


def _yes(flag: bool) -> str:
    return "YES" if flag else "no"


def _fmt_usd(amount: float) -> str:
    if amount <= 0:
        return "$0"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.3f}"


def _fields_summary(case: CaseResult) -> str:
    extraction = case.llm.extraction
    if extraction is None:
        return "-"
    collapsed = collapse_products(extraction.products)
    if collapsed is None:
        return "-"
    if isinstance(collapsed, list):
        return ", ".join(collapsed)
    return collapsed


def render_table(cases: list[CaseResult]) -> str:
    header = " ".join(name.upper().ljust(width) for name, width in _COLS)
    lines = [header, "-" * len(header)]
    for case in cases:
        extraction = case.llm.extraction
        row = [
            _fit(case.name, 22),
            _fit(_fields_summary(case), 22),
            _fit(extraction.quantity if extraction else "-", 4),
            _fit(
                extraction.requested_delivery_date if extraction else "-",
                10,
            ),
            _fit(_yes(case.decision.human_review_required), 3),
            _fit(_yes(case.decision.sales_team_flag), 3),
            _fit(f"{extraction.confidence:.2f}" if extraction else "-", 4),
            _fit(f"{case.llm.provider}/{case.llm.mode}", 16),
        ]
        lines.append(" ".join(row))
    return "\n".join(lines)


def render_details(cases: list[CaseResult]) -> str:
    lines = ["", "details " + "-" * 72]
    for case in cases:
        decision = case.decision
        lines.append(
            f"{case.name} HR={_yes(decision.human_review_required)} "
            f"ST={_yes(decision.sales_team_flag)}"
        )
        if decision.review_reason:
            lines.append(f"  human: {decision.review_reason}")
        if decision.sales_team_reason:
            lines.append(f"  sales: {decision.sales_team_reason}")
        if case.llm.total_tokens:
            lines.append(
                f"  tokens: {case.llm.prompt_tokens} in / "
                f"{case.llm.completion_tokens} out "
                f"(~{_fmt_usd(case.llm.cost_usd)})"
            )
    return "\n".join(lines)


def render_summary(cases: list[CaseResult], wall_s: float) -> str:
    n = len(cases)
    human = sum(1 for c in cases if c.decision.human_review_required)
    sales = sum(1 for c in cases if c.decision.sales_team_flag)
    latencies = [c.llm.latency_ms for c in cases if c.llm.ok]
    avg = int(sum(latencies) / len(latencies)) if latencies else 0
    prompt = sum(c.llm.prompt_tokens for c in cases)
    completion = sum(c.llm.completion_tokens for c in cases)
    tokens = sum(c.llm.total_tokens for c in cases) or (prompt + completion)
    usd = estimate_cost_usd(prompt, completion)
    return (
        f"\n{n} emails | human {human} | sales {sales} | "
        f"tokens {prompt} in / {completion} out ({tokens} tot) | "
        f"~{_fmt_usd(usd)} | avg LLM latency {avg}ms | wall {wall_s:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process order emails into results.json"
    )
    parser.add_argument("--dir", type=Path, default=EMAIL_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        default_providers()
    except RuntimeError as exc:
        print(f"[harness] {exc}", file=sys.stderr)
        sys.exit(1)

    processed_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    cases = run(args.dir, args.limit)
    write_results(cases, args.out, processed_at)
    print()
    print(render_table(cases))
    print(render_details(cases))
    print(render_summary(cases, time.perf_counter() - start))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
