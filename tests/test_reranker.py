"""
Unit tests for the cross-encoder reranker (src/rag/reranker.py).

These tests use simulate_reranking() so they run without downloading the
568 MB BAAI/bge-reranker-v2-m3 weights — fast and CI-friendly.

Integration tests (requiring the real model) are skipped unless
RERANKER_INTEGRATION_TESTS=1 is set in the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.rag.reranker import CrossEncoderReranker, RerankResult, simulate_reranking


# ─── Minimal stub for RetrievedChunk (avoids importing heavy deps) ────────────

@dataclass
class _Chunk:
    """Minimal stand-in for HybridRetriever.RetrievedChunk."""
    chunk_id: str
    content: str
    score: float = 0.0
    retrieval_method: str = "rrf"
    contract_id: str = "C-001"
    supplier_id: str = "SUP-001"
    page_number: int = 1
    is_table: bool = False
    metadata: dict = field(default_factory=dict)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def delay_chunks() -> list[_Chunk]:
    """
    Classic reranking failure scenario from the project:

    Query: "penalty for delays beyond 60 days"
    BM25/RRF problem: "Day 31 through 60" chunk contains the keyword "60"
    so keyword retrieval ranks it higher than the semantically correct
    "Day 61 and beyond" chunk.

    The cross-encoder should reverse this ordering.
    """
    return [
        _Chunk(
            chunk_id="chunk-day31-60",
            content="Late delivery penalties: Day 31 through 60: 1.5% of invoice value per week.",
            score=0.82,   # RRF ranked this first — keyword "60" collision
        ),
        _Chunk(
            chunk_id="chunk-day61-plus",
            content=(
                "Escalated penalties for extended delays — Day 61 and beyond: "
                "3.0% of invoice value per week, plus right to terminate contract."
            ),
            score=0.71,   # RRF ranked this second — actual answer
        ),
        _Chunk(
            chunk_id="chunk-grace",
            content="A 5-business-day grace period applies before any penalty accrues.",
            score=0.55,
        ),
        _Chunk(
            chunk_id="chunk-force-majeure",
            content="Penalties are waived during force majeure events as defined in Section 18.",
            score=0.40,
        ),
    ]


@pytest.fixture
def empty_chunks() -> list[_Chunk]:
    return []


# ─── simulate_reranking tests ─────────────────────────────────────────────────

class TestSimulateReranking:
    """Tests for the lightweight Jaccard-based simulation (no model required)."""

    def test_returns_list(self, delay_chunks):
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])
        assert isinstance(result, list)

    def test_length_preserved(self, delay_chunks):
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])
        assert len(result) == len(delay_chunks)

    def test_sorted_descending_by_cross_encoder_score(self, delay_chunks):
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])
        scores = [r["cross_encoder_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_cross_encoder_score_field_present(self, delay_chunks):
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])
        for r in result:
            assert "cross_encoder_score" in r
            assert "rrf_score" in r

    def test_rrf_score_preserved(self, delay_chunks):
        """Original RRF scores must be accessible after simulation."""
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])
        rrf_scores = {r["chunk_id"]: r["rrf_score"] for r in result}
        assert rrf_scores["chunk-day31-60"] == pytest.approx(0.82)
        assert rrf_scores["chunk-day61-plus"] == pytest.approx(0.71)

    def test_semantically_correct_chunk_promoted(self, delay_chunks):
        """
        Core correctness test: the 'Day 61+' chunk should score higher than
        'Day 31-60' for the query about 'beyond 60 days'.

        Jaccard overlap: 'Day 61 and beyond' shares tokens {penalty, day, beyond,
        delays, 60} more densely with the query than 'Day 31 through 60'.
        """
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])

        scores_by_id = {r["chunk_id"]: r["cross_encoder_score"] for r in result}
        # 'day61-plus' should outscore 'day31-60' because it contains "beyond"
        assert scores_by_id["chunk-day61-plus"] >= scores_by_id["chunk-day31-60"], (
            "Expected 'Day 61+' chunk to rank above 'Day 31-60' for query about "
            "'beyond 60 days', but simulate_reranking ranked them incorrectly."
        )

    def test_empty_input(self, empty_chunks):
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [])
        assert result == []

    def test_empty_content_chunk(self):
        chunks = [{"chunk_id": "c1", "content": "", "score": 0.5}]
        result = simulate_reranking("some query", chunks)
        assert len(result) == 1
        assert result[0]["cross_encoder_score"] == 0.0

    def test_score_range(self, delay_chunks):
        """Simulated scores should be in the [-3, 3] range."""
        query = "penalty for delays beyond 60 days"
        result = simulate_reranking(query, [c.__dict__ for c in delay_chunks])
        for r in result:
            assert -3.0 <= r["cross_encoder_score"] <= 3.0, (
                f"Score {r['cross_encoder_score']} out of expected [-3, 3] range"
            )


# ─── CrossEncoderReranker unit tests (no real model) ─────────────────────────

class TestCrossEncoderRerankerUnit:
    """Unit tests for class interface — model loading mocked out."""

    def test_default_model_name(self):
        r = CrossEncoderReranker()
        assert r.model_name == "BAAI/bge-reranker-v2-m3"

    def test_custom_model_name(self):
        r = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert r.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_not_loaded_at_init(self):
        r = CrossEncoderReranker()
        assert r._model is None

    def test_load_time_none_before_load(self):
        r = CrossEncoderReranker()
        assert r.load_time_s is None

    def test_empty_chunks_returns_empty(self, monkeypatch):
        """rerank() on empty list should return [] without touching the model."""
        r = CrossEncoderReranker()
        # Monkeypatch _ensure_loaded so it fails if called
        monkeypatch.setattr(r, "_ensure_loaded", lambda: (_ for _ in ()).throw(
            AssertionError("_ensure_loaded should NOT be called for empty input")
        ))
        result = r.rerank("some query", [])
        assert result == []

    def test_rerank_result_fields(self, monkeypatch, delay_chunks):
        """Check RerankResult has all expected fields."""
        r = CrossEncoderReranker()

        # Mock the model
        class _FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                # Return descending scores: first pair gets highest score
                return [1.5 - i * 0.3 for i in range(len(pairs))]

        r._model = _FakeModel()
        monkeypatch.setattr(r, "_ensure_loaded", lambda: None)

        results = r.rerank("penalty for delays beyond 60 days", delay_chunks, top_k=2)
        assert len(results) == 2
        for res in results:
            assert isinstance(res, RerankResult)
            assert hasattr(res, "chunk_id")
            assert hasattr(res, "content")
            assert hasattr(res, "cross_encoder_score")
            assert hasattr(res, "rrf_score")
            assert hasattr(res, "is_table")
            assert hasattr(res, "contract_id")

    def test_top_k_respected(self, monkeypatch, delay_chunks):
        r = CrossEncoderReranker()

        class _FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                return list(range(len(pairs), 0, -1))  # descending ints

        r._model = _FakeModel()
        monkeypatch.setattr(r, "_ensure_loaded", lambda: None)

        results = r.rerank("query", delay_chunks, top_k=2)
        assert len(results) == 2

    def test_top_k_larger_than_chunks(self, monkeypatch, delay_chunks):
        """top_k > len(chunks) should return all chunks, not error."""
        r = CrossEncoderReranker()

        class _FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                return [float(i) for i in range(len(pairs))]

        r._model = _FakeModel()
        monkeypatch.setattr(r, "_ensure_loaded", lambda: None)

        results = r.rerank("query", delay_chunks, top_k=100)
        assert len(results) == len(delay_chunks)

    def test_sorted_descending(self, monkeypatch, delay_chunks):
        """Results should always come back sorted high → low."""
        r = CrossEncoderReranker()

        class _FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                # Deliberately mixed scores
                return [0.3, 1.9, -0.5, 0.8]

        r._model = _FakeModel()
        monkeypatch.setattr(r, "_ensure_loaded", lambda: None)

        results = r.rerank("query", delay_chunks, top_k=4)
        scores = [r.cross_encoder_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_stored_in_result(self, monkeypatch, delay_chunks):
        """cross_encoder_score should differ from rrf_score after reranking."""
        r = CrossEncoderReranker()

        class _FakeModel:
            def predict(self, pairs, show_progress_bar=False):
                return [2.1, 0.4, 1.1, -0.2]

        r._model = _FakeModel()
        monkeypatch.setattr(r, "_ensure_loaded", lambda: None)

        results = r.rerank("query", delay_chunks, top_k=4)
        # rrf_score should match original chunk.score
        original_scores = {c.chunk_id: c.score for c in delay_chunks}
        for res in results:
            assert res.rrf_score == pytest.approx(original_scores[res.chunk_id])


# ─── Integration tests (skipped unless env var set) ───────────────────────────

INTEGRATION = pytest.mark.skipif(
    os.getenv("RERANKER_INTEGRATION_TESTS") != "1",
    reason="Set RERANKER_INTEGRATION_TESTS=1 to run (downloads 568 MB model)",
)


@INTEGRATION
class TestCrossEncoderRerankerIntegration:
    """Requires actual model download. Run with RERANKER_INTEGRATION_TESTS=1."""

    def test_model_loads(self):
        r = CrossEncoderReranker()
        r._ensure_loaded()
        assert r._model is not None
        assert r.load_time_s is not None and r.load_time_s > 0

    def test_reranking_promotes_correct_chunk(self, delay_chunks):
        """Real model should rank 'Day 61+' above 'Day 31-60'."""
        r = CrossEncoderReranker()
        query = "penalty for delays beyond 60 days"
        results = r.rerank(query, delay_chunks, top_k=2)
        assert results[0].chunk_id == "chunk-day61-plus", (
            f"Expected 'chunk-day61-plus' at rank 1, got '{results[0].chunk_id}'"
        )
