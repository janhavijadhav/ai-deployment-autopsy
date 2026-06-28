"""
Tests for the hybrid retriever: RRF fusion, table boost, cosine similarity.
"""

from __future__ import annotations

import pytest
from src.rag.retriever import HybridRetriever, RetrievedChunk


def make_chunk(chunk_id: str, score: float = 0.5, is_table: bool = False) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=f"Content for {chunk_id}",
        score=score,
        retrieval_method="dense",
        contract_id="CTR-001",
        supplier_id="SUP-001",
        page_number=1,
        is_table=is_table,
        metadata={},
    )


class TestRRFFusion:
    def test_rrf_boosts_chunks_appearing_in_both_lists(self):
        """A chunk ranked highly in BOTH dense and sparse should outscore chunks in one list only."""
        retriever = HybridRetriever()

        dense  = [make_chunk("A"), make_chunk("B"), make_chunk("C")]
        sparse = [make_chunk("A"), make_chunk("D"), make_chunk("E")]

        merged = retriever._reciprocal_rank_fusion(dense, sparse)
        merged.sort(key=lambda x: x.score, reverse=True)

        # "A" appears in both lists at rank 0 → highest RRF score
        assert merged[0].chunk_id == "A"

    def test_rrf_score_is_sum_of_reciprocals(self):
        """Manually verify RRF math: score(rank_i) = 1/(k + rank_i + 1)."""
        retriever = HybridRetriever()
        k = HybridRetriever.RRF_K

        dense  = [make_chunk("X")]   # rank 0 in dense
        sparse = [make_chunk("X")]   # rank 0 in sparse

        merged = retriever._reciprocal_rank_fusion(dense, sparse)
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)
        assert abs(merged[0].score - expected) < 1e-9

    def test_chunk_only_in_dense_has_lower_score_than_both(self):
        """A chunk in one list only scores lower than one appearing in both."""
        retriever = HybridRetriever()
        k = HybridRetriever.RRF_K

        dense  = [make_chunk("A"), make_chunk("B")]
        sparse = [make_chunk("A")]

        merged = retriever._reciprocal_rank_fusion(dense, sparse)
        score_map = {c.chunk_id: c.score for c in merged}

        # A is in both (rank 0 each) → 2/(k+1)
        # B is only in dense at rank 1 → 1/(k+2)
        assert score_map["A"] > score_map["B"]

    def test_all_input_chunks_appear_in_output(self):
        """No chunks should be silently dropped by RRF."""
        retriever = HybridRetriever()
        dense  = [make_chunk("A"), make_chunk("B")]
        sparse = [make_chunk("C"), make_chunk("D")]
        merged = retriever._reciprocal_rank_fusion(dense, sparse)
        ids = {c.chunk_id for c in merged}
        assert ids == {"A", "B", "C", "D"}


class TestTableBoost:
    def test_table_chunk_score_boosted(self):
        """Table chunks should have their score multiplied by TABLE_BOOST."""
        retriever = HybridRetriever()
        boost = HybridRetriever.TABLE_BOOST

        chunks = [
            make_chunk("table-1", score=0.5, is_table=True),
            make_chunk("text-1",  score=0.5, is_table=False),
        ]
        boosted = retriever._apply_table_boost(chunks)

        table_chunk = next(c for c in boosted if c.chunk_id == "table-1")
        text_chunk  = next(c for c in boosted if c.chunk_id == "text-1")

        assert abs(table_chunk.score - 0.5 * boost) < 1e-9
        assert abs(text_chunk.score - 0.5) < 1e-9

    def test_non_table_chunk_score_unchanged(self):
        """Text chunks must not be affected by the table boost."""
        retriever = HybridRetriever()
        chunks = [make_chunk("text-only", score=0.75, is_table=False)]
        boosted = retriever._apply_table_boost(chunks)
        assert abs(boosted[0].score - 0.75) < 1e-9

    def test_table_boost_constant_is_above_one(self):
        """TABLE_BOOST must be > 1.0 — otherwise it's not a boost."""
        assert HybridRetriever.TABLE_BOOST > 1.0

    def test_table_outranks_text_with_equal_rrf_score(self):
        """After boost, a table chunk with the same base score should rank above text."""
        retriever = HybridRetriever()
        chunks = [
            make_chunk("table", score=0.5, is_table=True),
            make_chunk("text",  score=0.5, is_table=False),
        ]
        boosted = retriever._apply_table_boost(chunks)
        boosted.sort(key=lambda c: c.score, reverse=True)
        assert boosted[0].chunk_id == "table"


class TestBM25IndexBuild:
    @pytest.mark.asyncio
    async def test_bm25_index_built_from_chunks(self):
        """After build_bm25_index, _bm25_index and _bm25_corpus are populated."""
        retriever = HybridRetriever()
        chunks = [
            make_chunk("c1"),
            make_chunk("c2"),
        ]
        # Mutate content for realistic BM25
        chunks[0].content = "late delivery penalty SLA clause"
        chunks[1].content = "payment terms net thirty days"

        await retriever.build_bm25_index(chunks)

        assert retriever._bm25_index is not None
        assert len(retriever._bm25_corpus) == 2

    @pytest.mark.asyncio
    async def test_sparse_search_returns_empty_before_index(self):
        """_sparse_search returns [] gracefully if index hasn't been built."""
        retriever = HybridRetriever()
        results = await retriever._sparse_search("penalty clause", None, 5)
        assert results == []
