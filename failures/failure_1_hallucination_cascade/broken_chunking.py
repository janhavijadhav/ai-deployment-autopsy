"""
BROKEN: Naive character-based chunker — causes Failure 1 (Hallucination Cascade).

This is the original chunker that was in production.
It produces faithfulness score of ~34% on contract queries involving tables.

DO NOT USE IN PRODUCTION. Kept for comparison and demo purposes.
See src/rag/chunking.py for the fixed TableAwareChunker.
"""

from __future__ import annotations


class BrokenNaiveChunker:
    """
    BROKEN chunker. Splits on character count with fixed overlap.
    Completely unaware of document structure, headings, or tables.

    The failure mode:
    - Contract PDFs contain pricing/penalty tables
    - Tables split mid-row at character boundary
    - LLM receives truncated table structure
    - LLM hallucinates the rest of the table
    - Users get wrong penalty amounts, prices, SLA terms

    Faithfulness score: ~34%
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "doc") -> list[dict]:
        """
        Split text into fixed-size character chunks.
        DOES NOT handle tables, headings, or semantic boundaries.
        """
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self.chunk_size
            content = text[start:end]

            chunks.append({
                "chunk_id": f"{doc_id}-{idx:04d}",
                "content": content,
                "chunk_type": "text",  # Everything is "text" — no table detection
                "start_char": start,
                "end_char": end,
                "doc_id": doc_id,
            })

            start = end - self.overlap  # Overlap to avoid losing context... except it still cuts tables
            idx += 1

        return chunks


def demonstrate_failure():
    """
    Show how naive chunking corrupts a contract penalty table.
    """
    contract_excerpt = """
SECTION 9.3 — LATE DELIVERY PENALTIES

If Supplier fails to deliver Goods or Services by the Delivery Date specified in the
applicable Purchase Order, Supplier shall pay to Buyer liquidated damages as follows:

| Delay Period      | Penalty Rate  | Maximum Cap     | Payment Due      |
|-------------------|---------------|-----------------|------------------|
| Day 1 through 30  | 0.5% per day  | 10% of PO Value | Net-30 from calc |
| Day 31 through 60 | 0.75% per day | 15% of PO Value | Net-30 from calc |
| Day 61 and beyond | 1.0% per day  | 20% of PO Value | Net-15 from calc |
| Force Majeure     | Waived        | N/A             | N/A              |

Penalties shall be calculated on the total value of the delayed line items, not the
total PO value. Supplier must notify Buyer within 24 hours of anticipated delay.
The 24-hour notification window serves as a grace period before penalties accrue.
"""

    chunker = BrokenNaiveChunker(chunk_size=400, overlap=50)
    chunks = chunker.chunk(contract_excerpt, doc_id="CTR-09871_apex")

    print("=" * 65)
    print("BROKEN CHUNKER — Failure 1 Demonstration")
    print("=" * 65)
    print(f"Input length: {len(contract_excerpt)} chars")
    print(f"Chunks produced: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} (chars {chunk['start_char']}–{chunk['end_char']}) ---")
        print(repr(chunk["content"][:120]) + "...")
        print()

    print(">>> PROBLEM:")
    print("    The penalty table is split across chunks 1 and 2.")
    print("    Chunk 1 ends mid-table: '| Day 31 through 60 | 0.75%'")
    print("    Chunk 2 starts:         'per day | 15% of PO...'")
    print()
    print("    If a user asks 'What are the Apex delivery penalties?'")
    print("    The retriever returns Chunk 1 (highest cosine sim).")
    print("    The LLM sees an incomplete table and hallucinates the rest.")
    print("    Faithfulness score: ~0.34")
    print()
    print(">>> FIX: Use TableAwareChunker in src/rag/chunking.py")
    print("    Tables become atomic chunks. Faithfulness score: ~0.91")


if __name__ == "__main__":
    demonstrate_failure()
