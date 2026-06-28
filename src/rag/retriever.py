"""
Hybrid retriever: Dense (BGE-M3 / Qdrant) + Sparse (BM25) with RRF fusion.

Using both dense and sparse retrieval closes a common enterprise RAG gap:
dense excels at semantic similarity, BM25 excels at exact-match term lookup
(e.g. supplier IDs like "SUP-0042", clause numbers like "Section 9.3.1").
Pure dense retrieval misses exact-match lookups; pure BM25 misses semantic ones.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, ScoredPoint
from rank_bm25 import BM25Okapi

from src.config import settings
from src.observability.tracing import tracer


# ─── Embedding model (singleton, loaded once) ────────────────────────────────

_bge_model: BGEM3FlagModel | None = None


def get_embedding_model() -> BGEM3FlagModel:
    global _bge_model
    if _bge_model is None:
        _bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cpu")
    return _bge_model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with BGE-M3. Returns dense vectors."""
    model = get_embedding_model()
    result = model.encode(
        texts,
        batch_size=12,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return result["dense_vecs"].tolist()


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float
    retrieval_method: str   # "dense", "sparse", or "rrf"
    contract_id: str
    supplier_id: str
    page_number: int
    is_table: bool
    metadata: dict[str, Any]


# ─── Hybrid Retriever ─────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Dense + sparse retrieval with Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)  where k=60 (standard)

    Table chunks get a 1.2× score boost since structured data queries
    (pricing, SLAs, delivery terms) should prefer table sources.
    """

    RRF_K = 60
    TABLE_BOOST = 1.25

    def __init__(self):
        self._qdrant: AsyncQdrantClient | None = None
        self._bm25_index: BM25Okapi | None = None
        self._bm25_corpus: list[RetrievedChunk] = []

    async def _get_qdrant(self) -> AsyncQdrantClient:
        if self._qdrant is None:
            self._qdrant = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
        return self._qdrant

    async def retrieve(
        self,
        query: str,
        supplier_id: str | None = None,
        top_k: int = 5,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """
        Full hybrid retrieval pipeline:
        1. Dense search via Qdrant (BGE-M3 vectors)
        2. BM25 keyword search over same corpus
        3. Merge results via RRF
        4. Apply table boost
        5. Return top_k
        """
        with tracer.span("retriever.hybrid") as span:
            span.set_attribute("query_len", len(query))
            span.set_attribute("supplier_filter", supplier_id or "none")

            # Parallel dense + sparse
            dense_results, sparse_results = await asyncio.gather(
                self._dense_search(query, supplier_id, dense_top_k),
                self._sparse_search(query, supplier_id, sparse_top_k),
            )

            merged = self._reciprocal_rank_fusion(dense_results, sparse_results)
            merged = self._apply_table_boost(merged)
            merged.sort(key=lambda x: x.score, reverse=True)

            span.set_attribute("dense_hits", len(dense_results))
            span.set_attribute("sparse_hits", len(sparse_results))
            span.set_attribute("merged_hits", len(merged))

            return merged[:top_k]

    async def _dense_search(
        self,
        query: str,
        supplier_id: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Search Qdrant with dense BGE-M3 embeddings."""
        client = await self._get_qdrant()
        query_vec = embed([query])[0]

        qdrant_filter = None
        if supplier_id:
            qdrant_filter = Filter(
                must=[FieldCondition(key="supplier_id", match=MatchValue(value=supplier_id))]
            )

        results: list[ScoredPoint] = await client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vec,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [
            RetrievedChunk(
                chunk_id=r.id,
                content=r.payload.get("content", ""),
                score=r.score,
                retrieval_method="dense",
                contract_id=r.payload.get("contract_id", ""),
                supplier_id=r.payload.get("supplier_id", ""),
                page_number=r.payload.get("page_number", 0),
                is_table=r.payload.get("is_table", False),
                metadata=r.payload,
            )
            for r in results
        ]

    async def _sparse_search(
        self,
        query: str,
        supplier_id: str | None,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """BM25 keyword search over in-memory corpus."""
        if not self._bm25_index or not self._bm25_corpus:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score <= 0:
                break
            chunk = self._bm25_corpus[idx]
            if supplier_id and chunk.supplier_id != supplier_id:
                continue
            results.append(RetrievedChunk(
                **{**chunk.__dict__, "score": float(score), "retrieval_method": "sparse"}
            ))
        return results

    def _reciprocal_rank_fusion(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Merge dense and sparse results using RRF."""
        scores: dict[str, float] = {}
        chunks: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (self.RRF_K + rank + 1)
            chunks[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(sparse):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (self.RRF_K + rank + 1)
            if chunk.chunk_id not in chunks:
                chunks[chunk.chunk_id] = chunk

        merged = []
        for chunk_id, rrf_score in scores.items():
            chunk = chunks[chunk_id]
            merged.append(RetrievedChunk(
                **{**chunk.__dict__, "score": rrf_score, "retrieval_method": "rrf"}
            ))
        return merged

    def _apply_table_boost(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Boost table chunks — structured queries should prefer table sources."""
        for chunk in chunks:
            if chunk.is_table:
                chunk.score *= self.TABLE_BOOST
        return chunks

    async def build_bm25_index(self, chunks: list[RetrievedChunk]) -> None:
        """Build in-memory BM25 index from ingested chunks."""
        self._bm25_corpus = chunks
        tokenized = [c.content.lower().split() for c in chunks]
        self._bm25_index = BM25Okapi(tokenized)


# ─── Singleton retriever ──────────────────────────────────────────────────────

_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
