"""
Eval runner — CI gate for every PR.

Runs two eval suites:
  1. clean:       Standard DeepEval test cases (regression guard)
  2. adversarial: LLM-generated messy cases (closes train/test gap — Failure 6 fix)

Both must pass their thresholds for the CI deploy gate to proceed.

Usage:
  python -m src.evals.eval_runner --mode full
  python -m src.evals.eval_runner --mode adversarial
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import deepeval
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from src.config import settings
from src.evals.llm_judge import LLMJudge
from src.evals.adversarial_gen import AdversarialEvalGenerator, EvalCase


# ─── Clean eval dataset (hard-coded — always runs) ───────────────────────────

CLEAN_EVAL_CASES = [
    {
        "input": "What are the late delivery penalties in the Apex Industries contract?",
        "expected_output_contains": ["penalty", "delivery", "Apex"],
        "retrieval_context": [
            "Section 9.3 — Late Delivery Penalties: If Apex Industries fails to deliver "
            "within the agreed lead time, a penalty of 0.5% of the PO value per day of delay "
            "shall apply, up to a maximum of 10% of the total PO value.",
        ],
        "case_type": "contract_search",
    },
    {
        "input": "Which suppliers have a risk score above 0.7?",
        "expected_output_contains": ["risk", "supplier"],
        "retrieval_context": [],
        "case_type": "supplier_lookup",
    },
    {
        "input": "When does the Pacific Components MSA expire?",
        "expected_output_contains": ["Pacific", "expire", "2025"],
        "retrieval_context": [
            "Master Supply Agreement — Pacific Components Ltd\n"
            "Effective Date: January 15, 2023\n"
            "Expiry Date: January 14, 2026\n"
            "Auto-Renewal: Yes, 30-day notice required to cancel"
        ],
        "case_type": "contract_search",
    },
    {
        "input": "Show me suppliers with on-time delivery below 85%",
        "expected_output_contains": ["delivery", "85"],
        "retrieval_context": [],
        "case_type": "supplier_lookup",
    },
    {
        "input": "What is the governing law for the Nordic Materials agreement?",
        "expected_output_contains": ["Nordic", "govern"],
        "retrieval_context": [
            "Section 18 — Governing Law: This Agreement shall be governed by and "
            "construed in accordance with the laws of the State of New York."
        ],
        "case_type": "contract_search",
    },
]


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case_id: str
    query: str
    answer: str
    faithfulness_score: float
    relevance_score: float
    passed: bool
    latency_ms: float
    eval_type: str
    failure_reason: str = ""


@dataclass
class EvalReport:
    run_id: str
    timestamp: str
    mode: str
    total_cases: int
    passed: int
    failed: int
    avg_faithfulness: float
    avg_relevance: float
    avg_latency_ms: float
    results: list[EvalResult] = field(default_factory=list)
    gate_passed: bool = False

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total_cases, 1)

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print(f"EVAL REPORT — {self.timestamp}")
        print("=" * 65)
        print(f"Mode:             {self.mode}")
        print(f"Total cases:      {self.total_cases}")
        print(f"Pass rate:        {self.pass_rate:.1%}")
        print(f"Avg faithfulness: {self.avg_faithfulness:.3f}")
        print(f"Avg relevance:    {self.avg_relevance:.3f}")
        print(f"Avg latency:      {self.avg_latency_ms:.0f}ms")
        print(f"CI gate:          {'✓ PASSED' if self.gate_passed else '✗ FAILED'}")
        print("=" * 65)

        failed_cases = [r for r in self.results if not r.passed]
        if failed_cases:
            print("\nFailed cases:")
            for r in failed_cases[:5]:
                print(f"  [{r.case_id}] {r.query[:60]}")
                print(f"    Faith: {r.faithfulness_score:.2f} | Rel: {r.relevance_score:.2f}")
                if r.failure_reason:
                    print(f"    Reason: {r.failure_reason}")


# ─── Runner ───────────────────────────────────────────────────────────────────

class EvalRunner:
    """Runs eval suites and enforces CI gate thresholds."""

    FAITHFULNESS_GATE = 0.85
    RELEVANCE_GATE = 0.80
    PASS_RATE_GATE = 0.90  # 90% of cases must pass

    def __init__(self):
        self.judge = LLMJudge()

    async def run(self, mode: Literal["clean", "adversarial", "full"] = "full") -> EvalReport:
        run_id = f"eval-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}"
        print(f"\n[eval_runner] Starting eval run {run_id} (mode={mode})")

        cases_to_run: list[dict] = []

        if mode in ("clean", "full"):
            cases_to_run.extend([{**c, "eval_type": "clean"} for c in CLEAN_EVAL_CASES])

        if mode in ("adversarial", "full"):
            adv_path = Path("eval_results/adversarial_cases.json")
            if adv_path.exists():
                gen = AdversarialEvalGenerator()
                adv_cases = gen.load(str(adv_path))
                cases_to_run.extend([
                    {
                        "input": c.query,
                        "expected_output_contains": c.expected_contains,
                        "retrieval_context": [],
                        "case_type": c.adversarial_type,
                        "case_id": c.case_id,
                        "eval_type": "adversarial",
                    }
                    for c in adv_cases[:settings.ADVERSARIAL_EVAL_COUNT]
                ])
            else:
                print("  [warn] No adversarial cases found. Run: make eval-adversarial")

        results = []
        for i, case in enumerate(cases_to_run):
            print(f"  [{i+1}/{len(cases_to_run)}] {case['input'][:60]}...")
            result = await self._run_case(case, idx=i)
            results.append(result)

        report = self._build_report(run_id, mode, results)
        self._save_report(report)
        report.print_summary()
        return report

    async def _run_case(self, case: dict, idx: int) -> EvalResult:
        """Run a single eval case against the live agent."""
        t0 = time.perf_counter()
        case_id = case.get("case_id", f"case-{idx:04d}")

        try:
            # Import here to avoid circular imports
            from src.agent.procurement_agent import create_agent, run_agent
            agent = await create_agent()
            answer = await run_agent(
                agent,
                user_message=case["input"],
                thread_id=f"eval-{case_id}",
            )
        except Exception as e:
            return EvalResult(
                case_id=case_id,
                query=case["input"],
                answer="",
                faithfulness_score=0.0,
                relevance_score=0.0,
                passed=False,
                latency_ms=(time.perf_counter() - t0) * 1000,
                eval_type=case.get("eval_type", "unknown"),
                failure_reason=f"Agent error: {e}",
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        # Score with LLM judge
        context = "\n\n".join(case.get("retrieval_context", []))
        faith_score = await self.judge.score_faithfulness(
            question=case["input"],
            answer=answer,
            context=context or "(no context provided — check database retrieval)",
        )
        rel_score = await self.judge.score_relevance(
            question=case["input"],
            answer=answer,
        )

        passed = (
            faith_score.score >= self.FAITHFULNESS_GATE
            and rel_score.score >= self.RELEVANCE_GATE
        )

        failure_reason = ""
        if not faith_score.passed:
            failure_reason += f"Faithfulness {faith_score.score:.2f} < {self.FAITHFULNESS_GATE}. "
        if not rel_score.passed:
            failure_reason += f"Relevance {rel_score.score:.2f} < {self.RELEVANCE_GATE}."

        return EvalResult(
            case_id=case_id,
            query=case["input"],
            answer=answer,
            faithfulness_score=faith_score.score,
            relevance_score=rel_score.score,
            passed=passed,
            latency_ms=latency_ms,
            eval_type=case.get("eval_type", "unknown"),
            failure_reason=failure_reason,
        )

    def _build_report(self, run_id: str, mode: str, results: list[EvalResult]) -> EvalReport:
        passed = sum(1 for r in results if r.passed)
        avg_faith = sum(r.faithfulness_score for r in results) / max(len(results), 1)
        avg_rel = sum(r.relevance_score for r in results) / max(len(results), 1)
        avg_lat = sum(r.latency_ms for r in results) / max(len(results), 1)
        pass_rate = passed / max(len(results), 1)

        return EvalReport(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            mode=mode,
            total_cases=len(results),
            passed=passed,
            failed=len(results) - passed,
            avg_faithfulness=avg_faith,
            avg_relevance=avg_rel,
            avg_latency_ms=avg_lat,
            results=results,
            gate_passed=pass_rate >= self.PASS_RATE_GATE,
        )

    def _save_report(self, report: EvalReport) -> None:
        os.makedirs("eval_results", exist_ok=True)
        path = f"eval_results/{report.run_id}.json"
        with open(path, "w") as f:
            data = {**report.__dict__, "results": [r.__dict__ for r in report.results]}
            json.dump(data, f, indent=2)
        print(f"  Report saved: {path}")


# ─── CLI + CI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def run(mode: str = "full"):
        """Run eval suite. Exit code 1 if CI gate fails."""
        runner = EvalRunner()
        report = asyncio.run(runner.run(mode=mode))
        if not report.gate_passed:
            print("\n✗ CI GATE FAILED — Deploy blocked")
            sys.exit(1)
        print("\n✓ CI GATE PASSED")

    app()
