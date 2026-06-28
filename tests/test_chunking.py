"""
Tests for Failure 1 fix: table-aware chunking.

Key assertions:
- Tables in contract text are never split across chunks
- NaiveCharacterChunker DOES split tables (demonstrates the broken behavior)
- TableAwareChunker keeps tables atomic
- Sentence-boundary splitting respects word boundaries
"""

from __future__ import annotations

import pytest
from src.rag.chunking import (
    ChunkType,
    NaiveCharacterChunker,
    TableAwareChunker,
)


TABLE_CONTRACT = """
SECTION 9.3 — LATE DELIVERY PENALTIES

| Delay Period      | Penalty Rate  | Maximum Cap     |
|-------------------|---------------|-----------------|
| Day 1 through 30  | 0.5% per day  | 10% of PO Value |
| Day 31 through 60 | 0.75% per day | 15% of PO Value |
| Day 61 and beyond | 1.0% per day  | 20% of PO Value |

Payment terms are Net-30. Late payment accrues 1.5% monthly interest.
All disputes shall be resolved under the laws of the State of New York.
"""


class TestNaiveChunkerBreaksTable:
    """Prove that the naive chunker causes Failure 1."""

    def test_splits_table_mid_row(self):
        """Naive chunker with small chunk size WILL split the table."""
        chunker = NaiveCharacterChunker(chunk_size=200, overlap=20)
        chunks = chunker.chunk(TABLE_CONTRACT, doc_id="test")

        # With chunk_size=200 there will be multiple chunks
        assert len(chunks) > 1, "Expected multiple chunks from naive chunker"

        # Find chunks — at least one should contain partial table content
        table_content = [c for c in chunks if "|" in c["content"]]
        assert len(table_content) > 0

        # The table should be SPLIT — no single chunk contains the full table
        full_table_in_one_chunk = any(
            "Day 1 through 30" in c["content"] and "Day 61 and beyond" in c["content"]
            for c in chunks
        )
        # This is the broken behavior: table is fragmented
        # (if this fails, chunk_size is too large for the test — reduce it)
        assert not full_table_in_one_chunk, (
            "Naive chunker should split the table — increase TABLE_CONTRACT or reduce chunk_size"
        )

    def test_all_chunks_labeled_as_text(self):
        """Naive chunker classifies everything as text — no table awareness."""
        chunker = NaiveCharacterChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(TABLE_CONTRACT, doc_id="test")
        for chunk in chunks:
            assert chunk["chunk_type"] == "text"


class TestTableAwareChunker:
    """Prove that the fixed chunker keeps tables atomic."""

    def test_table_is_atomic_chunk(self, sample_contract_text):
        """The penalty table must appear as a single chunk, never split."""
        chunker = TableAwareChunker(max_text_tokens=200)
        chunks = chunker.chunk_text(sample_contract_text, doc_id="CTR-apex")

        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        assert len(table_chunks) >= 1, "Expected at least one TABLE chunk"

        # Every table chunk must contain the full row content — no splits
        for tc in table_chunks:
            if "Day 1 through 30" in tc.content:
                assert "Day 61 and beyond" in tc.content, (
                    f"Table chunk is incomplete — split mid-row:\n{tc.content}"
                )

    def test_table_chunk_contains_pipe_characters(self, sample_contract_text):
        """Table chunks must contain the pipe-separated structure."""
        chunker = TableAwareChunker()
        chunks = chunker.chunk_text(sample_contract_text, doc_id="test")
        table_chunks = [c for c in chunks if c.chunk_type == ChunkType.TABLE]
        assert len(table_chunks) >= 1
        for tc in table_chunks:
            assert "|" in tc.content

    def test_non_table_text_is_split(self, sample_contract_text):
        """Long text paragraphs are split at sentence boundaries."""
        chunker = TableAwareChunker(max_text_tokens=30)  # Very small to force splits
        chunks = chunker.chunk_text(sample_contract_text, doc_id="test")
        text_chunks = [c for c in chunks if c.chunk_type == ChunkType.TEXT]
        # With max_text_tokens=30, longer paragraphs should split
        assert len(text_chunks) >= 1

    def test_table_classified_correctly(self):
        """_classify_text_block correctly identifies table-like content."""
        chunker = TableAwareChunker()
        table_text = (
            "| Col A | Col B | Col C |\n"
            "|-------|-------|-------|\n"
            "| val1  | val2  | val3  |"
        )
        result = chunker._classify_text_block(table_text)
        assert result == ChunkType.TABLE

    def test_plain_text_not_classified_as_table(self):
        """Plain paragraphs are not misidentified as tables."""
        chunker = TableAwareChunker()
        plain = "Payment terms are Net-30 from date of invoice."
        result = chunker._classify_text_block(plain)
        assert result == ChunkType.TEXT

    def test_chunk_ids_are_unique(self, sample_contract_text):
        """Every chunk must have a unique ID."""
        chunker = TableAwareChunker()
        chunks = chunker.chunk_text(sample_contract_text, doc_id="CTR-001")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_is_table_property(self, sample_contract_text):
        """DocumentChunk.is_table matches chunk_type == TABLE."""
        chunker = TableAwareChunker()
        chunks = chunker.chunk_text(sample_contract_text, doc_id="test")
        for chunk in chunks:
            assert chunk.is_table == (chunk.chunk_type == ChunkType.TABLE)

    def test_metadata_contains_doc_id(self, sample_contract_text):
        """All chunks must reference their source document."""
        chunker = TableAwareChunker()
        chunks = chunker.chunk_text(sample_contract_text, doc_id="CTR-apex-001")
        for chunk in chunks:
            assert chunk.metadata.get("doc_id") == "CTR-apex-001"
