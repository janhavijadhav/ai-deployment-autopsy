# Failure 2: The Latency Wall

> **Production symptom:** The procurement agent took 12–16 seconds to respond to queries.
> The enterprise customer's SLA requirement was < 3 seconds. They escalated after day 1.
> "This is slower than running a manual SAP report. What's the point?"

---

## What the Symptom Looked Like

```
[LangFuse trace — wall clock times]

00:00.000  user query received
00:01.240  ── search_contracts started
00:03.820  ── search_contracts done (2.58s)
00:03.820  ── lookup_supplier started (SEQUENTIAL — waited for contracts)
00:06.110  ── lookup_supplier done (2.29s)
00:06.110  ── flag_supplier_risks started (SEQUENTIAL — waited for supplier)
00:08.950  ── flag_supplier_risks done (2.84s)
00:08.950  ── LLM synthesis started
00:13.400  ── LLM synthesis done (4.45s)

Total: 13.4 seconds
```

---

## The Wrong Diagnosis

- "The LLM is slow — try GPT-4o instead" (LLM was only 4.4s of 13.4s total)
- "Our servers are too small — provision more compute"
- "Qdrant is the bottleneck" (it was sub-100ms)

The profiling told a different story. 9 seconds of wall time was tool calls.
But each tool was individually fast (2–3s each). The issue wasn't speed — it was order.

---

## Actual Root Cause: Three separate problems

### 1. Sequential tool calls (biggest impact: −8s)

The agent was calling tools one at a time:

```python
# BROKEN — sequential, wasteful
contracts = await search_contracts(query)       # 2.5s
supplier  = await lookup_supplier(supplier_id)  # 2.3s — waited for nothing
risks     = await flag_risks([supplier_id])     # 2.8s — waited for nothing

# All three tools are INDEPENDENT — they don't need each other's output
```

### 2. No caching (−2s on repeat/similar queries)

Embeddings were recomputed from scratch on every request. No Redis cache.
A procurement analyst asking the same contract question twice paid full price twice.

### 3. Embeddings recomputed at query time

BGE-M3 embedding for the user query took 400ms on CPU, every single request.
Pre-warming the embedding model at startup (not cold-loading on first request)
removed this latency spike.

---

## The Fix

### 1. Async parallel tool calls with `asyncio.gather`

```python
# FIXED — parallel execution
contracts, supplier, risks = await asyncio.gather(
    search_contracts(query),
    lookup_supplier(supplier_id),
    flag_risks([supplier_id]),
)
# Wall time: max(2.5s, 2.3s, 2.8s) = 2.8s — not 7.6s
```

See: `src/agent/tools.py` — all tools are `async def`
See: `src/agent/procurement_agent.py` — LangGraph ToolNode runs parallel calls

### 2. Redis semantic cache

```python
# Check cache before tool call (src/cache/redis_cache.py)
if cached := await cache.get(namespace, query):
    return cached   # 5ms — not 2.5s
```

Semantic similarity threshold: 0.92 cosine similarity.
"What are Apex's SLA terms?" hits cache for "What SLAs apply to Apex Industries?"

### 3. Pre-compute embeddings + warm model at startup

```python
# src/rag/retriever.py — singleton loaded at import time
_bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
# First request no longer pays the 400ms model load penalty
```

---

## Before / After Metrics

| Metric | Before | After |
|--------|--------|-------|
| p50 latency | 13.4s | 780ms |
| p95 latency | 18.2s | 1.4s |
| p99 latency | 22.1s | 2.1s |
| Cache hit rate (steady state) | 0% | ~38% |
| Tool parallelism | sequential | 3-way parallel |
| LLM fraction of total | 33% | 72% |

The LLM is now the majority of latency — meaning we've optimized everything else.
Further gains require a faster model or prompt compression.

---

## Grafana Dashboard Evidence

The before/after is visible in the `procurement_agent_llm_latency_ms` and
`procurement_agent_tool_latency_ms` histograms in Grafana.

The p95 drop from 18.2s → 1.4s showed up immediately after deploying the fix
and is the clearest evidence that sequential tool calls — not LLM speed or
infrastructure — was the root cause.
