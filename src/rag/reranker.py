"""
Cross-Encoder Reranker
======================
Third stage of the hybrid RAG pipeline:

  Dense (BGE-M3) ──┐
                    ├── RRF Fusion ──→ Table Boost ──→ Cross-Encoder Rerank ──→ top-k
  Sparse (BM25)  ──┘

Why cross-encoding beats bi-encoding for reranking
----------------------------------------------------
Bi-encoders (BGE-M3) embed query and document independently — fast but the
representations never "see" each other. BM25 relies purely on keyword overlap.

A cross-encoder takes the (query, document) pair as a single input, letting
every query token attend to every document token. This catches cases where
BM25 ranks "Day 31 through 60" above "Day 61 and beyond" for the query
"penalty for delays beyond 60 days" — a keyword collision the cross-encoder
easily resolves.

Trade-off: O(k) forward passes per query (one per candidate). We run it only
on the top-N candidates from RRF (default N = top_k × 4), keeping latency
bounded while maximising precision.

Model choice
------------
BAAI/bge-reranker-v2-m3 pairs naturally with BAAI/bge-m3 embeddings:
- Multilingual (100+ languages — matches adversarial eval Failure 6)
- 568 MB — fits comfortably in a single A10 GPU or CPU inference
- Significantly outperforms ms-marco-MiniLM on domain-specific text

Singleton pattern
-----------------
Model loading takes ~4s and ~1.5 GB RAM. We load once per process via
get_reranker() and reuse across all requests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# Lazy import — sentence-transformers is a heavy dep; don't import at module load
_reranker_instance: "CrossEncoderReranker | None" = None


@dataclass
class RerankResult:
    """Scored chunk after cross-encoder reranking."""
    chunk_id: str
    content: str
    cross_encoder_score: float      # Raw logit from cross-encoder (higher = more relevant)
    rrf_score: float                 # Original RRF score before reranking
    is_table: bool
    contract_id: str
    supplier_id: str
    page_number: int
    metadata: dict = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.cross_encoder_score


class CrossEncoderReranker:
    """
    Reranks a candidate list using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, rrf_chunks, top_k=5)

    The reranker accepts RetrievedChunk objects from HybridRetriever and
    returns RerankResult objects sorted by cross-encoder score descending.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
    CANDIDATE_MULTIPLIER = 4        # Retrieve this many × top_k candidates before reranking
    MAX_SEQ_LENGTH = 512            # Truncate to this many tokens per document

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_seq_length: int = MAX_SEQ_LENGTH,
    ):
        self.model_name = model_name
        self.device = device
        self.max_seq_length = max_seq_length
        self._model = None
        self._load_time_s: float | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "pip install sentence-transformers>=3.0.0"
            ) from exc

        logger.info("Loading cross-encoder model: %s", self.model_name)
        t0 = time.perf_counter()
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_seq_length,
            device=self.device,
        )
        self._load_time_s = time.perf_counter() - t0
        logger.info("Cross-encoder loaded in %.2fs", self._load_time_s)

    def rerank(
        self,
        query: str,
        chunks: list,            # list[RetrievedChunk] — avoid circular import
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Score every (query, chunk) pair and return top_k by cross-encoder score.

        Parameters
        ----------
        query  : The user query string.
        chunks : Candidates from RRF — typically top_k × CANDIDATE_MULTIPLIER items.
        top_k  : How many to return after reranking.

        Returns
        -------
        List of RerankResult sorted by cross_encoder_score descending.
        """
        if not chunks:
            return []

        self._ensure_loaded()

        pairs = [(query, chunk.content) for chunk in chunks]

        t0 = time.perf_counter()
        scores = self._model.predict(pairs, show_progress_bar=False)
        latency_ms = (time.perf_counter() - t0) * 1000

        logger.debug(
            "Cross-encoder scored %d pairs in %.1fms (model: %s)",
            len(pairs), latency_ms, self.model_name,
        )

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        results: list[RerankResult] = []
        for chunk, score in scored[:top_k]:
            results.append(RerankResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                cross_encoder_score=float(score),
                rrf_score=chunk.score,
                is_table=chunk.is_table,
                contract_id=chunk.contract_id,
                supplier_id=chunk.supplier_id,
                page_number=chunk.page_number,
                metadata=chunk.metadata,
            ))

        return results

    async def rerank_async(
        self,
        query: str,
        chunks: list,
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Async wrapper — runs cross-encoder in a thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.rerank, query, chunks, top_k)

    @property
    def load_time_s(self) -> float | None:
        return self._load_time_s


# ── Singleton accessor ────────────────────────────────────────────────────────

def get_reranker(model_name: str = CrossEncoderReranker.DEFAULT_MODEL) -> CrossEncoderReranker:
    """
    Return the global CrossEncoderReranker singleton, creating it on first call.
    Model loading is deferred until the first rerank() call.
    """
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReranker(model_name=model_name)
    return _reranker_instance


# ── Lightweight demo (no model required) ─────────────────────────────────────

def simulate_reranking(
    query: str,
    chunks: list[dict],
) -> list[dict]:
    """
    Deterministic simulation of cross-encoder reranking for demos and tests
    that don't have the full model loaded.

    Scores each chunk by computing rough semantic overlap between the query
    and chunk content using token-level Jaccard similarity — good enough to
    demonstrate the reranking effect without 568 MB of model weights.
    """
    import re

    def tokenize(text: str) -> set[str]:
        return set(re.findall(r'\b\w+\b', text.lower()))

    query_tokens = tokenize(query)

    scored = []
    for chunk in chunks:
        chunk_tokens = tokenize(chunk.get("content", ""))
        if not chunk_tokens:
            score = 0.0
        else:
            intersection = query_tokens & chunk_tokens
            union = query_tokens | chunk_tokens
            jaccard = len(intersection) / len(union)
            # Cross-encoders tend to score in [-5, 5] range; scale Jaccard to [-3, 3]
            score = (jaccard * 6.0) - 3.0

        scored.append({**chunk, "cross_encoder_score": round(score, 4), "rrf_score": chunk.get("score", 0.0)})

    return sorted(scored, key=lambda x: x["cross_encoder_score"], reverse=True)
