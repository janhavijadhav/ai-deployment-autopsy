"""
Adversarial eval generator — the fix for Failure 6 (The Eval That Lied).

THE PROBLEM:
  Offline eval dataset had 96% accuracy. Production had 61%.
  Root cause: eval dataset was too clean.
  It contained only:
    - Perfect grammar, correctly spelled supplier names
    - Unambiguous single-supplier queries
    - English-only questions
    - Simple fact retrieval (not multi-hop)

  Real procurement analysts write queries like:
    - "wat r the penaltis for apex if they deliver late" (typos)
    - "supplier 42 sla terms" (ID-style reference)
    - "show me risky chinese vendors" (informal, filter-heavy)
    - "was the delivery penalty clause changed in the 2023 amendment?" (multi-hop)
    - "Quelles sont les conditions de paiement d'Apex Industries?" (French)

THE FIX:
  Use Claude as an adversarial attacker to generate:
    1. Typo-injected variants of clean queries
    2. Ambiguous queries (supplier name partially wrong)
    3. Multi-language queries (French, German, Spanish, Mandarin romanized)
    4. Multi-hop queries requiring contract+supplier cross-reference
    5. Adversarially ambiguous queries (two valid answers)

  Close the train/test distribution gap. If the adversarial eval passes,
  production accuracy is within 5% of offline accuracy.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from typing import Literal

from anthropic import AsyncAnthropic

from src.config import settings


@dataclass
class EvalCase:
    case_id: str
    query: str
    expected_supplier_ids: list[str]
    expected_contains: list[str]    # Substrings that must appear in the answer
    adversarial_type: str
    difficulty: Literal["easy", "medium", "hard"]
    notes: str = ""


ADVERSARIAL_GENERATOR_PROMPT = """You are an adversarial test case generator for an enterprise procurement AI.

Your job is to create REALISTIC MESSY queries that procurement analysts actually type.
These should expose weaknesses in the AI system that clean test cases miss.

Given these clean base queries about a manufacturing company's procurement system:
{base_queries}

Generate {count} adversarial variants. Include a mix of:
- Typo-injected queries (misspelled supplier names, transposed letters, wrong spaces)
- Ambiguous queries (supplier name partially wrong, two suppliers match)
- Multi-language queries (French, German, Spanish — don't translate, just rephrase naturally)
- Multi-hop queries (require cross-referencing contract AND supplier records)
- Informal/slang queries ("show me the sketchy chinese vendors", "apex late delivery fine?")
- Queries with wrong assumptions ("What are apex's net-60 terms?" when they have net-30)
- Mobile-style truncated queries ("sla penalt apex 2023")

For each case, specify:
- query: the adversarial query text
- adversarial_type: one of [typo, ambiguous, multilingual, multi_hop, informal, wrong_assumption, truncated]
- difficulty: easy/medium/hard
- expected_contains: list of substrings that a correct answer must contain

Return a JSON array of objects with these fields. Return ONLY valid JSON.
"""

# Clean base queries — these pass the clean eval at 96%
BASE_QUERIES = [
    "What are the penalty clauses for late delivery in the Apex Industries contract?",
    "Show me all high-risk suppliers in China",
    "What is the contract value for supplier SUP-0007?",
    "When does the Apex Industries MSA expire and does it auto-renew?",
    "Which suppliers have on-time delivery below 85%?",
    "What are the payment terms for supplier SUP-0015?",
    "Flag all suppliers with risk score above 0.7",
    "What contracts expire in the next 90 days?",
    "What is the governing law for the Pacific Components agreement?",
    "Summarize risk exposure in our electronics category",
]


class AdversarialEvalGenerator:
    """Generate adversarial test cases using LLM-as-attacker."""

    def __init__(self):
        self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def generate(self, count: int = 50) -> list[EvalCase]:
        """Generate adversarial eval cases via Claude attacker."""
        prompt = ADVERSARIAL_GENERATOR_PROMPT.format(
            base_queries=json.dumps(BASE_QUERIES, indent=2),
            count=count,
        )

        response = await self._client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]

        raw_cases = json.loads(text)

        cases = []
        for i, raw in enumerate(raw_cases):
            cases.append(EvalCase(
                case_id=f"adv-{i:04d}",
                query=raw["query"],
                expected_supplier_ids=raw.get("expected_supplier_ids", []),
                expected_contains=raw.get("expected_contains", []),
                adversarial_type=raw.get("adversarial_type", "unknown"),
                difficulty=raw.get("difficulty", "medium"),
                notes=raw.get("notes", ""),
            ))

        return cases

    def save(self, cases: list[EvalCase], path: str = "eval_results/adversarial_cases.json") -> None:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump([c.__dict__ for c in cases], f, indent=2)
        print(f"Saved {len(cases)} adversarial cases to {path}")

    def load(self, path: str = "eval_results/adversarial_cases.json") -> list[EvalCase]:
        with open(path) as f:
            raw = json.load(f)
        return [EvalCase(**r) for r in raw]


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import typer

    app = typer.Typer()

    @app.command()
    def generate(count: int = 50, output: str = "eval_results/adversarial_cases.json"):
        """Generate adversarial eval cases."""
        gen = AdversarialEvalGenerator()
        cases = asyncio.run(gen.generate(count=count))
        gen.save(cases, path=output)
        typer.echo(f"Generated {len(cases)} adversarial cases")
        # Print type breakdown
        from collections import Counter
        types = Counter(c.adversarial_type for c in cases)
        for t, n in types.most_common():
            typer.echo(f"  {t}: {n}")

    app()
