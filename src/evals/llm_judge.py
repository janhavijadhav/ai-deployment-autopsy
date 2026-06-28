"""
LLM-as-judge faithfulness and relevance scoring.

Used as part of the Failure 1 fix (checking hallucination rate)
and the Failure 6 fix (closing the eval distribution gap).

Faithfulness: does the answer contain only claims supported by the retrieved context?
  Before fix (naive chunking): 34%
  After fix (table-aware chunking): 91%

Relevance: does the answer address the user's actual question?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from anthropic import AsyncAnthropic

from src.config import settings


@dataclass
class JudgeScore:
    score: float               # 0.0 – 1.0
    label: Literal["pass", "fail"]
    reasoning: str
    metric: str
    threshold: float

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


# ─── Faithfulness Judge ───────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are a strict faithfulness evaluator for an enterprise procurement AI.

Your job: determine whether the ANSWER is fully grounded in the CONTEXT.
A faithful answer contains only claims that are directly supported by the context.
Hallucination includes: inventing supplier names, contract terms, prices, dates, or IDs
that do not appear in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Evaluate faithfulness on a scale from 0.0 to 1.0:
- 1.0: Every claim in the answer is explicitly supported by the context
- 0.7–0.9: Mostly grounded with minor interpretive steps
- 0.4–0.7: Mix of grounded and ungrounded claims
- 0.0–0.4: Significant hallucination present

Respond with valid JSON only:
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<one paragraph explaining your score>",
  "hallucinated_claims": ["<claim 1>", "<claim 2>"]
}}
"""

RELEVANCE_PROMPT = """You are an answer relevance evaluator for an enterprise procurement AI.

Evaluate whether the ANSWER actually addresses what the QUESTION is asking.
Irrelevant answers include: responses about a different supplier, different contract clause,
different metric, or general information when a specific answer was requested.

QUESTION:
{question}

ANSWER:
{answer}

Score answer relevance from 0.0 to 1.0:
- 1.0: Directly and completely answers the question
- 0.7–0.9: Addresses the question with minor gaps
- 0.4–0.7: Partially addresses the question
- 0.0–0.4: Does not address the question

Respond with valid JSON only:
{{
  "score": <float 0.0–1.0>,
  "reasoning": "<one paragraph explaining your score>",
  "missing_elements": ["<element 1>", "<element 2>"]
}}
"""


class LLMJudge:
    """Claude-as-judge for faithfulness and relevance evaluation."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.CLAUDE_MODEL
        self._client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    async def score_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
        threshold: float | None = None,
    ) -> JudgeScore:
        """Score whether the answer is grounded in retrieved context."""
        threshold = threshold or settings.EVAL_FAITHFULNESS_THRESHOLD
        prompt = FAITHFULNESS_PROMPT.format(
            context=context[:4000],  # Keep within context window
            question=question,
            answer=answer,
        )
        raw = await self._call_judge(prompt)
        return JudgeScore(
            score=raw.get("score", 0.0),
            label="pass" if raw.get("score", 0.0) >= threshold else "fail",
            reasoning=raw.get("reasoning", ""),
            metric="faithfulness",
            threshold=threshold,
        )

    async def score_relevance(
        self,
        question: str,
        answer: str,
        threshold: float | None = None,
    ) -> JudgeScore:
        """Score whether the answer addresses the question."""
        threshold = threshold or settings.EVAL_RELEVANCY_THRESHOLD
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        raw = await self._call_judge(prompt)
        return JudgeScore(
            score=raw.get("score", 0.0),
            label="pass" if raw.get("score", 0.0) >= threshold else "fail",
            reasoning=raw.get("reasoning", ""),
            metric="relevance",
            threshold=threshold,
        )

    async def _call_judge(self, prompt: str) -> dict:
        """Call Claude as judge and parse JSON response."""
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
