# Verder Liquids — order intake

Structured extraction of customer order emails. The LLM fills fields; Python
decides routing. Pricing is out of scope. Missing values stay `null`.

**Thesis: the LLM structures data; deterministic Python decides.**

`human_review_required` and `sales_team_flag` are independent. Both can be
true. Human means the order cannot proceed until someone intervenes. Sales
means the order can proceed or needs parallel sales attention (quote, price,
model selection).

## Layout

```
.
├── schema.py             Extraction contract and results.json shape
├── prompts.py            System prompt; email body is untrusted DATA
├── rules.py              decide() — no LLM calls
├── llm.py                Structured extraction with a degradation ladder
├── harness.py            Batch runner → results.json
├── results.json          Last live run
├── REAL_Test case/       Brief, 7 emails, output schema, datasheets
└── tests/
    ├── test_schema.py
    ├── test_rules.py
    └── test_llm_ladder.py
```

Datasheet PDFs are used only as a filename list of known Packo codes
(`ICP2`, `MCP2`, `MCP3`, `MWP2`, `NMS`, `PHP2`, `PRP2`). The agent does not
read the PDFs and does not pick a model from a datasheet.

## Setup

Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put API keys in `.env`. Never commit `.env`. Offline tests do not need keys.

Kill switch for tracing: `LANGSMITH_TRACING=false`. If LangSmith is down, the
run still completes. Raw email logs go to `logs/` (gitignored).

## Run

```bash
source .venv/bin/activate
python tests/test_schema.py
python tests/test_rules.py
python tests/test_llm_ladder.py
python harness.py
```

`python llm.py` probes whether strict structured output is accepted.
`python llm.py path/to/email.txt` extracts a single file.

## Pipeline

1. `extract_email()` — OpenAI structured output (`ExtractedEmail`).
2. On a strict-schema rejection, the same provider retries in JSON mode.
3. On transport failure, hop to OpenRouter (same SDK, different base URL).
4. Invalid JSON and model refusals are terminal: the case goes to a human.
5. `decide(extraction, raw_email)` sets the two flags. The LLM never sets them.

Email body is data, never instructions. Prompt-injection markers are scanned
in Python.

## Output

One object per email, matching `REAL_Test case/03_Output_schema.json`:

- `source_file` — original filename (spaces and `#` kept)
- `extraction` — fields or `null`; `product_reference` is string, array, or null
- `order_processing.human_review_required` / `review_reason`
- `routing.sales_team_flag` / `sales_team_reason`

## Stack

- OpenAI Python SDK 2.x + Pydantic structured output
- Pure functions for business rules
- LangSmith optional (`@traceable` on `extract_email`)
- No LangChain chains, no document RAG

## Security

- `.env` is gitignored. Only `.env.example` is tracked.
- `logs/` is gitignored (raw email is personal data).
- Invalid JSON never reaches `decide()`.
- Customer email is personal data under GDPR. This prototype may use LangSmith
  cloud; production should evaluate EU-hosted observability.
