.PHONY: help up down seed ingest eval lint test clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─── Infrastructure ─────────────────────────────────────────────────────────────
up:  ## Spin up full stack (agent + observability)
	docker compose up -d
	@echo "✓ Stack running. Grafana: http://localhost:3001  LangFuse: http://localhost:3000"

down:  ## Tear down stack
	docker compose down -v

logs:  ## Stream agent logs
	docker compose logs -f agent

# ─── Data ───────────────────────────────────────────────────────────────────────
seed:  ## Seed fake SAP data into DuckDB + SQLite
	python -m data.seed_data

ingest:  ## Ingest contracts into Qdrant
	python -m src.rag.pipeline ingest --contracts-dir data/contracts/

# ─── Evals ──────────────────────────────────────────────────────────────────────
eval:  ## Run full eval suite (CI gate)
	python -m src.evals.eval_runner --mode full

eval-adversarial:  ## Generate adversarial test cases + run
	python -m src.evals.adversarial_gen --count 50
	python -m src.evals.eval_runner --mode adversarial

eval-faithfulness:  ## LLM-as-judge faithfulness check
	python -m src.evals.llm_judge --check faithfulness

# ─── Schema Monitor ─────────────────────────────────────────────────────────────
schema-snapshot:  ## Take schema snapshot (run before migrations)
	python -m src.data.schema_monitor snapshot

schema-diff:  ## Detect schema drift vs last snapshot
	python -m src.data.schema_monitor diff

# ─── Dev ────────────────────────────────────────────────────────────────────────
lint:  ## Lint + format
	ruff check src/ tests/ --fix
	ruff format src/ tests/

test:  ## Run unit tests
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:  ## Remove generated artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	rm -rf .coverage htmlcov eval_results/

# ─── Failure Mode Demos ─────────────────────────────────────────────────────────
demo-failure-1:  ## Run hallucination cascade demo (broken then fixed)
	python failures/failure_1_hallucination_cascade/demo.py

demo-failure-4:  ## Trigger schema drift bomb
	python failures/failure_4_schema_drift_bomb/trigger_migration.py
	python -m src.data.schema_monitor diff
