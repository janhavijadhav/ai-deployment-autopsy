"""
RAG ingest + retrieval pipeline.

Ingest: PDF → table-aware chunks → BGE-M3 embeddings → Qdrant + BM25 index
Query:  question → hybrid retrieval → reranked chunks
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.config import settings
from src.observability.tracing import tracer
from src.rag.chunking import TableAwareChunker, DocumentChunk
from src.rag.retriever import embed, get_retriever, RetrievedChunk


VECTOR_DIM = 1024  # BGE-M3 dense vector dimension


# ─── Ingest ───────────────────────────────────────────────────────────────────

async def ingest_contracts(contracts_dir: str = "data/contracts/") -> dict[str, int]:
    """
    Ingest all PDF contracts from a directory into Qdrant + build BM25 index.

    Returns a dict of {filename: chunks_ingested}.
    """
    chunker = TableAwareChunker()
    client = AsyncQdrantClient(url=settings.QDRANT_URL)

    # Ensure collection exists
    await _ensure_collection(client)

    stats: dict[str, int] = {}
    all_chunks: list[RetrievedChunk] = []

    for pdf_path in Path(contracts_dir).glob("*.pdf"):
        doc_id = pdf_path.stem
        with tracer.span("ingest.pdf", attributes={"doc_id": doc_id}):
            chunks = chunker.chunk_pdf(str(pdf_path), doc_id=doc_id)
            stats[pdf_path.name] = len(chunks)

            # Extract supplier/contract metadata from filename convention
            # Convention: {contract_id}_{supplier_id}.pdf
            parts = doc_id.split("_", 1)
            contract_id = parts[0] if parts else doc_id
            supplier_id = parts[1] if len(parts) > 1 else "UNKNOWN"

            points = []
            for chunk in chunks:
                vector = embed([chunk.content])[0]
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "content": chunk.content,
                        "contract_id": contract_id,
                        "supplier_id": supplier_id,
                        "page_number": chunk.page_number,
                        "is_table": chunk.is_table,
                        "chunk_type": chunk.chunk_type.value,
                        **chunk.metadata,
                    },
                )
                points.append(point)
                # Build parallel list for BM25
                all_chunks.append(RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    score=0.0,
                    retrieval_method="none",
                    contract_id=contract_id,
                    supplier_id=supplier_id,
                    page_number=chunk.page_number,
                    is_table=chunk.is_table,
                    metadata=chunk.metadata,
                ))

            # Batch upsert to Qdrant
            await client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points,
                wait=True,
            )

    # Build BM25 index over all ingested chunks
    retriever = get_retriever()
    await retriever.build_bm25_index(all_chunks)

    return stats


async def retrieve_contracts(
    query: str,
    supplier_id: str | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval entry point (called by the agent tool).
    Returns serialisable dicts for JSON tool response.
    """
    retriever = get_retriever()
    chunks = await retriever.retrieve(query, supplier_id=supplier_id, top_k=top_k)
    return [
        {
            "chunk_id": c.chunk_id,
            "contract_id": c.contract_id,
            "supplier_id": c.supplier_id,
            "content": c.content,
            "score": round(c.score, 4),
            "is_table": c.is_table,
            "page_number": c.page_number,
            "retrieval_method": c.retrieval_method,
        }
        for c in chunks
    ]


# ─── Helpers ──────────────────────────────────────────────────────────────────

async def _ensure_collection(client: AsyncQdrantClient) -> None:
    """Create Qdrant collection if it doesn't exist."""
    existing = await client.get_collections()
    names = [c.name for c in existing.collections]
    if settings.QDRANT_COLLECTION not in names:
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


# ─── CLI entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import sys
    import typer

    app = typer.Typer()

    @app.command()
    def ingest(contracts_dir: str = "data/contracts/"):
        """Ingest PDF contracts into Qdrant."""
        stats = asyncio.run(ingest_contracts(contracts_dir))
        for fname, count in stats.items():
            typer.echo(f"  {fname}: {count} chunks")
        typer.echo(f"Done. Total chunks: {sum(stats.values())}")

    app()
