"""LLM wrapper: structured extraction with a degradation ladder.

OpenAI SDK + Pydantic. No langchain imports here.

Ladder for every email (first success wins):

  1. primary provider, strict structured output  -> chat.completions.parse()
  2. same provider, JSON mode + Pydantic on the way back
     (only after a strict-schema rejection)
  3. fallback provider (OpenRouter), same two steps

Design decisions:
  - Transport errors (timeout, 429, 5xx) retry with backoff, then hop provider.
  - A strict-schema 400 is remembered for the session: that provider drops to
    JSON mode and never pays the failing call again.
  - ValidationError and model refusals are TERMINAL on purpose. No repair loop,
    no provider hop: a broken contract is a model-quality signal, not a
    transport failure. The raw output is logged and the harness routes the
    case to human review. Nothing invalid ever reaches decide().
  - Tracing (LangSmith) is optional and fail-silent. Kill switch:
    LANGSMITH_TRACING=false. A tracing outage must never take down a run.
  - Raw + parsed are appended to logs/calls.jsonl on every outcome.
    logs/ is gitignored (raw email is personal data).

    python llm.py             # probe: per provider, strict vs JSON mode
    python llm.py email.txt   # extract one file and print the result
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from prompts import build_system_prompt, wrap_email
from schema import ExtractedEmail

load_dotenv()

# ---------------------------------------------------------------------------
# Config (env-tunable; defaults match .env.example)
# ---------------------------------------------------------------------------

PRIMARY_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-terra")
FALLBACK_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-5.6-terra")
FALLBACK_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
REQUEST_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "45"))
LOG_PATH = Path(os.getenv("LLM_LOG_PATH", "logs/calls.jsonl"))
_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "none")
# gpt-5.6-terra short-context list price (USD / 1M tokens). Override in .env.
INPUT_USD_PER_1M = float(os.getenv("LLM_INPUT_USD_PER_1M", "2.0"))
OUTPUT_USD_PER_1M = float(os.getenv("LLM_OUTPUT_USD_PER_1M", "12.0"))


def _sampler_kwargs() -> dict[str, Any]:
    """Some models reject sampler params. Set LLM_TEMPERATURE=none to drop it."""
    if _TEMPERATURE.strip().lower() in {"", "none", "default"}:
        return {}
    return {"temperature": float(_TEMPERATURE)}


# ---------------------------------------------------------------------------
# Tracing: optional, fail-silent. LANGSMITH_TRACING=false is the kill switch.
# ---------------------------------------------------------------------------

TRACING = os.getenv("LANGSMITH_TRACING", "false").strip().lower() == "true"


def _noop_traceable(*args: Any, **kwargs: Any):
    """Stand-in for langsmith.traceable. Supports bare and parametrized use."""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]

    def deco(fn):
        return fn

    return deco


traceable = _noop_traceable
if TRACING:
    try:
        from langsmith import traceable as _ls_traceable

        traceable = _ls_traceable
    except Exception as exc:  # tracing must never block a run
        print(
            f"[llm] tracing requested but langsmith unavailable ({exc}); "
            "continuing without tracing",
            file=sys.stderr,
        )
# wrap_openai is unused on purpose. It serializes chat.completions.parse()
# and warns because message.parsed is ExtractedEmail, not None.
# @traceable on extract_email is enough for LangSmith.


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    name: str
    client: Any
    model: str
    strict_ok: bool = True  # flips off for the session after a schema rejection


def _make_client(api_key: str, base_url: str | None = None) -> Any:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT_S)


_default_cache: list[Provider] | None = None


def default_providers(refresh: bool = False) -> list[Provider]:
    """Primary OpenAI, fallback OpenRouter. Cached so strict_ok persists."""
    global _default_cache
    if _default_cache is not None and not refresh:
        return _default_cache
    providers: list[Provider] = []
    if os.getenv("OPENAI_API_KEY"):
        providers.append(
            Provider("openai", _make_client(os.environ["OPENAI_API_KEY"]), PRIMARY_MODEL)
        )
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append(
            Provider(
                "openrouter",
                _make_client(os.environ["OPENROUTER_API_KEY"], FALLBACK_BASE_URL),
                FALLBACK_MODEL,
            )
        )
    if not providers:
        raise RuntimeError(
            "No provider configured. Fill OPENAI_API_KEY and/or "
            "OPENROUTER_API_KEY in .env (see .env.example)."
        )
    _default_cache = providers
    return providers


# ---------------------------------------------------------------------------
# Raw API calls. Retry covers transport only; schema errors surface at once.
# ---------------------------------------------------------------------------

_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

_retry_transient = retry(
    reraise=True,
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(_TRANSIENT),
)


@_retry_transient
def _call_strict(provider: Provider, system: str, user: str) -> Any:
    return provider.client.chat.completions.parse(
        model=provider.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=ExtractedEmail,
        **_sampler_kwargs(),
    )


def _json_mode_system(system: str) -> str:
    """JSON mode gets the schema in-prompt so it stays in sync with schema.py."""
    schema_json = json.dumps(ExtractedEmail.model_json_schema(), ensure_ascii=False)
    return (
        system
        + "\n\nReturn a single JSON object and nothing else."
        + " It must validate against this JSON Schema:\n"
        + schema_json
    )


@_retry_transient
def _call_json_mode(provider: Provider, system: str, user: str) -> Any:
    return provider.client.chat.completions.create(
        model=provider.model,
        messages=[
            {"role": "system", "content": _json_mode_system(system)},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        **_sampler_kwargs(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCHEMA_REJECTION_MARKERS = (
    "invalid schema",
    "response_format",
    "json_schema",
    "not permitted",
    "additionalproperties",
    "'required'",
)


def _is_schema_rejection(exc: BadRequestError) -> bool:
    """Heuristic. A false positive only costs dropping to JSON mode."""
    text = str(exc).lower()
    return any(marker in text for marker in _SCHEMA_REJECTION_MARKERS)


def _salvage_json(text: str) -> str:
    """Fallback-path insurance: some providers fence or pad JSON-mode output.

    The strict path never needs this. If salvage still fails, Pydantic raises
    ValidationError and the case goes to human review with the raw logged.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _token_usage(completion: Any) -> tuple[int, int, int]:
    """prompt, completion, total. Missing usage (fakes, hops) is 0, 0, 0."""
    usage = getattr(completion, "usage", None)
    if usage is None and isinstance(completion, dict):
        usage = completion.get("usage")
    if not usage:
        return 0, 0, 0
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        output = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        total = usage.get("total_tokens") or (int(prompt) + int(output))
        return int(prompt), int(output), int(total)
    prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    output = (
        getattr(usage, "completion_tokens", None)
        or getattr(usage, "output_tokens", 0)
        or 0
    )
    total = getattr(usage, "total_tokens", None) or (int(prompt) + int(output))
    return int(prompt), int(output), int(total)


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000) * INPUT_USD_PER_1M + (
        completion_tokens / 1_000_000
    ) * OUTPUT_USD_PER_1M


# ---------------------------------------------------------------------------
# Result + logging
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    ok: bool
    extraction: ExtractedEmail | None
    raw: str
    provider: str
    model: str
    mode: str  # "strict" | "json_object" | "none"
    error: str | None
    latency_ms: int
    attempts: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return estimate_cost_usd(self.prompt_tokens, self.completion_tokens)


def _log_call(email: str, result: ExtractionResult) -> None:
    """Raw + parsed always, per the briefing. Logging must never crash a run."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ok": result.ok,
            "provider": result.provider,
            "model": result.model,
            "mode": result.mode,
            "latency_ms": result.latency_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "cost_usd": round(result.cost_usd, 6),
            "attempts": result.attempts,
            "error": result.error,
            "email": email,
            "raw": result.raw,
            "parsed": (
                result.extraction.model_dump(mode="json") if result.extraction else None
            ),
        }
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[llm] could not write {LOG_PATH}: {exc}", file=sys.stderr)


def _finish(email: str, result: ExtractionResult) -> ExtractionResult:
    _log_call(email, result)
    return result


# ---------------------------------------------------------------------------
# The extraction function
# ---------------------------------------------------------------------------

@traceable(name="extract_email")
def extract_email(
    raw_email: str, providers: list[Provider] | None = None
) -> ExtractionResult:
    """Run the ladder. Never raises on provider or model behaviour.

    The harness turns ok=False into a human_review Decision, so a failed
    extraction degrades to a queued case, never to a crashed run.
    """
    provs = providers if providers is not None else default_providers()
    system = build_system_prompt()
    user = wrap_email(raw_email)
    attempts: list[str] = []
    last_error: str | None = None

    for provider in provs:
        modes = (["strict"] if provider.strict_ok else []) + ["json_object"]
        for mode in modes:
            start = time.perf_counter()
            raw = ""
            prompt_tokens = completion_tokens = total_tokens = 0
            try:
                if mode == "strict":
                    completion = _call_strict(provider, system, user)
                    prompt_tokens, completion_tokens, total_tokens = _token_usage(
                        completion
                    )
                    message = completion.choices[0].message
                    refusal = getattr(message, "refusal", None)
                    if refusal:
                        # Terminal: a refusal on a B2B email means a human
                        # should read it, not another model.
                        attempts.append(f"{provider.name}/strict: model refusal")
                        return _finish(
                            raw_email,
                            ExtractionResult(
                                ok=False,
                                extraction=None,
                                raw=str(refusal),
                                provider=provider.name,
                                model=provider.model,
                                mode=mode,
                                error=f"model refusal: {refusal}",
                                latency_ms=_ms(start),
                                attempts=list(attempts),
                                prompt_tokens=prompt_tokens,
                                completion_tokens=completion_tokens,
                                total_tokens=total_tokens,
                            ),
                        )
                    extraction = getattr(message, "parsed", None)
                    raw = message.content or ""
                    if extraction is None:
                        extraction = ExtractedEmail.model_validate_json(
                            _salvage_json(raw)
                        )
                    elif not isinstance(extraction, ExtractedEmail):
                        extraction = ExtractedEmail.model_validate(extraction)
                    if not raw:
                        raw = extraction.model_dump_json()
                else:
                    completion = _call_json_mode(provider, system, user)
                    prompt_tokens, completion_tokens, total_tokens = _token_usage(
                        completion
                    )
                    raw = completion.choices[0].message.content or ""
                    extraction = ExtractedEmail.model_validate_json(_salvage_json(raw))

                attempts.append(f"{provider.name}/{mode}: ok")
                return _finish(
                    raw_email,
                    ExtractionResult(
                        ok=True,
                        extraction=extraction,
                        raw=raw,
                        provider=provider.name,
                        model=provider.model,
                        mode=mode,
                        error=None,
                        latency_ms=_ms(start),
                        attempts=list(attempts),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    ),
                )

            except ValidationError as exc:
                # Terminal by design: no repair loop, no provider hop. The raw
                # output is preserved for the human reviewer and the eval set.
                attempts.append(f"{provider.name}/{mode}: contract broken")
                return _finish(
                    raw_email,
                    ExtractionResult(
                        ok=False,
                        extraction=None,
                        raw=raw,
                        provider=provider.name,
                        model=provider.model,
                        mode=mode,
                        error=f"schema validation failed: {exc.errors()[:3]}",
                        latency_ms=_ms(start),
                        attempts=list(attempts),
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    ),
                )

            except BadRequestError as exc:
                if mode == "strict" and _is_schema_rejection(exc):
                    provider.strict_ok = False  # remember for the whole session
                    attempts.append(
                        f"{provider.name}/strict: schema rejected -> json mode"
                    )
                    last_error = f"{provider.name} strict rejection: {exc}"
                    continue
                attempts.append(f"{provider.name}/{mode}: bad request")
                last_error = f"{provider.name} bad request: {exc}"
                break  # same provider will keep rejecting; hop

            except _TRANSIENT as exc:
                attempts.append(
                    f"{provider.name}/{mode}: {type(exc).__name__} after retries"
                )
                last_error = f"{provider.name} {type(exc).__name__}: {exc}"
                break  # transport is down for this provider; hop

            except APIStatusError as exc:
                status = getattr(exc, "status_code", "?")
                attempts.append(f"{provider.name}/{mode}: HTTP {status}")
                last_error = f"{provider.name} HTTP {status}: {exc}"
                break  # auth or other hard HTTP error; hop

            except Exception as exc:  # unknown failure must not kill the batch
                attempts.append(f"{provider.name}/{mode}: {type(exc).__name__}")
                last_error = f"{provider.name} {type(exc).__name__}: {exc}"
                break

    return _finish(
        raw_email,
        ExtractionResult(
            ok=False,
            extraction=None,
            raw="",
            provider="none",
            model="none",
            mode="none",
            error=last_error or "no provider succeeded",
            latency_ms=0,
            attempts=list(attempts),
        ),
    )


# ---------------------------------------------------------------------------
# Probe: does strict mode accept ExtractedEmail on each provider?
# ---------------------------------------------------------------------------

_PROBE_EMAIL = (
    "Subject: Quotation request\n\n"
    "Dear Verder team, please quote 2 Verderflex Dura 45 hose pumps for "
    "sodium hypochlorite dosing. Delivery to Rotterdam by 15 September.\n"
    "Kind regards, J. Bakker, AquaChem BV"
)


def probe(providers: list[Provider] | None = None) -> None:
    """Print whether strict structured output is accepted on each provider."""
    provs = providers if providers is not None else default_providers()
    system = build_system_prompt()
    user = wrap_email(_PROBE_EMAIL)
    for provider in provs:
        print(f"\n== {provider.name} ({provider.model}) ==")
        try:
            completion = _call_strict(provider, system, user)
            parsed = getattr(completion.choices[0].message, "parsed", None)
            products = ", ".join(parsed.products) if parsed and parsed.products else "?"
            print(f"  strict structured output: OK (products={products})")
        except BadRequestError as exc:
            if _is_schema_rejection(exc):
                print("  strict structured output: SCHEMA REJECTED -> ladder will use JSON mode")
                print(f"    {exc}")
            else:
                print(f"  strict structured output: bad request: {exc}")
        except Exception as exc:
            print(f"  strict structured output: {type(exc).__name__}: {exc}")
        try:
            completion = _call_json_mode(provider, system, user)
            raw = completion.choices[0].message.content or ""
            extraction = ExtractedEmail.model_validate_json(_salvage_json(raw))
            print(f"  json mode + pydantic:     OK (products={extraction.products})")
        except Exception as exc:
            print(f"  json mode + pydantic:     {type(exc).__name__}: {exc}")
    print("\nProbe done. If strict is OK on the primary, the ladder never leaves step 1.")


def _main(argv: list[str]) -> None:
    if len(argv) > 1:
        text = Path(argv[1]).read_text(encoding="utf-8")
        result = extract_email(text)
        print(
            f"ok={result.ok} provider={result.provider} mode={result.mode} "
            f"latency={result.latency_ms}ms "
            f"tokens={result.prompt_tokens}/{result.completion_tokens} "
            f"~${result.cost_usd:.4f}"
        )
        print("attempts:", "; ".join(result.attempts))
        if result.extraction is not None:
            print(result.extraction.model_dump_json(indent=2))
        else:
            print("error:", result.error)
        return
    probe()


if __name__ == "__main__":
    _main(sys.argv)
