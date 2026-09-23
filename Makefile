# Local development entry points. Every target assumes the virtualenv at .venv.

PY := .venv/bin/python
BASELINE := eval/results/baseline.json
AFTER := eval/results/after.json

.PHONY: help venv up down data seed ingest test lint api mcp eval-baseline eval-after eval-30 eval-full smoke-baseline preflight smoke smoke-eval smoke-gate gate perf image

help:
	@grep -E '^[a-z-]+:.*## ' Makefile | sed 's/:.*## /\t/'

venv: ## Create the virtualenv and install everything
	uv venv --python 3.11 .venv
	VIRTUAL_ENV=.venv uv pip install -e ".[dev,eval]"

up: ## Start local Qdrant and Postgres containers
	docker compose up -d

down: ## Stop the local containers (data volumes are kept)
	docker compose down

data: ## Regenerate the synthetic catalog (deterministic)
	$(PY) scripts/generate_data.py

seed: ## Load the catalog into Postgres for the MCP tools
	$(PY) -m qc_copilot.tools.seed

ingest: ## Rebuild the Qdrant knowledge base
	$(PY) -m qc_copilot.ingest.pipeline --recreate

test: ## Run the unit test suite
	$(PY) -m pytest -q

lint: ## Lint and check formatting
	$(PY) -m ruff check src tests scripts
	$(PY) -m ruff format --check src tests scripts

api: ## Serve the API on http://localhost:8088
	.venv/bin/uvicorn qc_copilot.api.main:app --host 0.0.0.0 --port 8088 --reload

mcp: ## Run the MCP tool server on stdio (for inspectors)
	$(PY) -m qc_copilot.mcp_server.server

GOLDEN_FULL := eval/golden/golden_set.jsonl
GOLDEN_30 := eval/golden/eval30.jsonl
GOLDEN_SMOKE := eval/golden/smoke.jsonl
GOLDEN ?= $(GOLDEN_FULL)

eval-baseline: ## Retrieval-only baseline on $(GOLDEN); resumes from its checkpoint if interrupted
	$(PY) -m qc_copilot.evaluation.run --mode retrieval --label baseline --golden $(GOLDEN) --out $(BASELINE)

eval-after: ## Agentic pipeline with guardrails on $(GOLDEN); resumes from its checkpoint if interrupted
	$(PY) -m qc_copilot.evaluation.run --mode agentic --label agentic_guarded --golden $(GOLDEN) --out $(AFTER)

eval-30: ## Baseline and after on the fixed 30-row stratified subset (reuses full-run checkpoints)
	$(MAKE) eval-baseline GOLDEN=$(GOLDEN_30)
	$(MAKE) eval-after GOLDEN=$(GOLDEN_30)

eval-full: ## Baseline and after on all 60 rows
	$(MAKE) eval-baseline GOLDEN=$(GOLDEN_FULL)
	$(MAKE) eval-after GOLDEN=$(GOLDEN_FULL)

smoke-baseline: ## Reference smoke result the CI gate compares against (agentic, 10 rows)
	$(PY) -m qc_copilot.evaluation.run --mode agentic --label smoke_baseline --golden $(GOLDEN_SMOKE) --out eval/results/smoke_baseline.json

preflight: ## One authenticated request per configured provider; fails on a rejected key
	$(PY) -m qc_copilot.llm.preflight --require groq gemini

smoke: smoke-eval smoke-gate ## What CI runs on every push: agentic smoke eval gated against smoke_baseline

smoke-eval: ## Collect and judge the 10-row smoke set (exit 2 when every provider is capped)
	$(PY) -m qc_copilot.evaluation.run --mode agentic --label smoke_ci --golden $(GOLDEN_SMOKE) --out eval/results/smoke_ci.json --fresh

smoke-gate: ## Compare the smoke result against the committed smoke baseline (exit 3 = warning, answer model differs)
	$(PY) -m qc_copilot.evaluation.gate --baseline eval/results/smoke_baseline.json --candidate eval/results/smoke_ci.json --margin 0.05 --tool-success-from-baseline --enforce-only-on-baseline-model

gate: ## After must stay within 0.05 of baseline faithfulness and not regress on tool success
	$(PY) -m qc_copilot.evaluation.gate --baseline $(BASELINE) --candidate $(AFTER) --margin 0.05 --tool-success-from-baseline

perf: ## Record p95 latency and cost per 1k queries into reports/perf.json
	$(PY) -m qc_copilot.evaluation.perf $(BASELINE)
	$(PY) -m qc_copilot.evaluation.perf $(AFTER)

image: ## Build the deployment image locally
	docker build -t quickcommerce-copilot:local .
