# AI Deployment Autopsy

> **I intentionally broke this system 6 different ways — then documented how to diagnose and fix each one. Because real engineering lives on the other side of deployment.**

Everyone's portfolio shows a system they built. This shows systems I **broke deliberately**,
**diagnosed systematically**, and **fixed in production**. That's the actual job.

**[→ Live Interactive Demo](https://ai-deployment-autopsy.streamlit.app)** — inject each failure mode and watch detection + remediation in real time.

---

## What This Is

A live, instrumented enterprise AI deployment for a fictitious Fortune 500 manufacturer —
**Meridian Manufacturing Corp** — running a multi-agent procurement intelligence platform
across 1,200+ supplier contracts and $2.4B in annual spend.

The platform uses:
- **LangGraph** for stateful multi-agent orchestration
- **Claude Sonnet** (Anthropic) as the reasoning LLM
- **Groq** (Llama 3.3 70B) for the live agent chat demo
- **Qdrant + BGE-M3 + BM25** for hybrid RAG over contract PDFs
- **DuckDB + SQLite** as a mock SAP data layer
- **Redis** for semantic caching
- **LangFuse + Prometheus + Grafana** for full observability
- **DeepEval + LLM-as-judge** for evaluation

Then I injected **6 production failure modes** — each representing a real class of enterprise
AI breakage — and wrote the postmortem for each one.

---

## Architecture

```
              ┌─────────────────────────────────────────────┐
              │       Procurement Intelligence Platform      │
              │     LangGraph Supervisor + 3 Specialists     │
              │    Groq (Llama 3.3 70B) · Claude Sonnet      │
              └──────────┬───────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼───────┐ ┌──────▼──────┐ ┌──────▼──────────┐
│  RAG Pipeline  │ │  Data Layer  │ │  Supplier API   │
│  BGE-M3 + BM25 │ │  DuckDB ·   │ │  OAuth2 client  │
│  RRF Fusion    │ │  SQLite ·   │ │  Mutex refresh  │
│  Cross-Encoder │ │  Schema Mon  │ └─────────────────┘
└───────┬────────┘ └─────────────┘
        │
┌───────▼────────────────────────────────────────────┐
│         Redis Semantic Cache  ·  38% hit rate       │
└───────┬────────────────────────────────────────────┘
        │
┌───────▼────────────────────────────────────────────┐
│      LangFuse · Prometheus · Grafana · PagerDuty    │
└────────────────────────────────────────────────────┘
```

---

## The 6 Failures

Each is a real failure class — wrong diagnosis included, because that's the part that costs the most time.

---

### 1 · Hallucination Cascade

**The symptom:** Agent gave wrong penalty amounts. An analyst nearly approved a supplier waiver based on a fabricated "2% per week." The actual contract said "0.5% per day" — a 10× difference.

**Everyone assumed:** LLM confabulation. Tried different models, lower temperature. Didn't help.

**What actually happened:** Naive character-based chunking was splitting penalty tables across chunk boundaries. The retriever returned half a table. The LLM hallucinated the rest based on what penalty clauses typically look like.

**The fix:** Table-aware chunking — tables are atomic chunks, never split. 1.25× retrieval score boost for structured data.

| Metric | Before | After |
|--------|--------|-------|
| Faithfulness score | 34% | 91% |
| Table recall | 12% | 94% |
| Hallucinated prices | 8/10 queries | 0/10 |

→ [`failures/failure_1_hallucination_cascade/`](failures/failure_1_hallucination_cascade/README.md) · Fix: [`src/rag/chunking.py`](src/rag/chunking.py)

---

### 2 · Latency Wall

**The symptom:** 12–16 seconds per query. Enterprise SLA was < 3 seconds. Day 1 feedback: *"This is slower than running a manual SAP report."*

**Everyone assumed:** LLM is slow. Infrastructure is undersized. (Profiling showed: LLM was 4.5s of 13.4s total. Qdrant was sub-100ms.)

**What actually happened:** Three compounding issues — sequential tool calls that blocked each other, no caching so embeddings recomputed every request, and BGE-M3 cold-loaded on first use.

```python
# Before: 7.7s of avoidable waiting
contracts = await search_contracts(query)   # 2.58s
supplier  = await lookup_supplier(sid)      # 2.29s  ← blocked for no reason
risks     = await flag_risks([sid])         # 2.84s  ← blocked for no reason

# After: 2.84s max (longest task wins)
contracts, supplier, risks = await asyncio.gather(
    search_contracts(query),
    lookup_supplier(sid),
    flag_risks([sid]),
)
```

**The fix:** Async parallel tool execution + Redis semantic cache + pre-warmed BGE-M3 singleton.

| Metric | Before | After |
|--------|--------|-------|
| p50 latency | 13.4s | 780ms |
| p95 latency | 18.2s | 1.4s |
| Cache hit rate | 0% | ~38% |

→ [`failures/failure_2_latency_wall/`](failures/failure_2_latency_wall/README.md)

---

### 3 · Context Collapse

**The symptom:** Multi-step contract approval workflows failed at step 3 — every time. The agent asked users to re-state the contract ID as if the conversation had just started.

**Everyone assumed:** LLM not following prompt. Session expiry. Network issue.

**What actually happened:** The code kept only the last 6 messages. Approval workflows generate 7+ messages by turn 3. The truncation silently dropped the `approval_id` and `contract_id` from turn 1. The agent had no memory an approval had been initiated.

**The fix:** LangGraph `AsyncSqliteSaver` checkpointer persists approval state to SQLite. Intelligent summarization replaces truncation — old turns are compressed, preserving all IDs, amounts, and chain status.

| Metric | Before | After |
|--------|--------|-------|
| Workflow completion | 0% (fails at turn 3) | 97% |
| State survives restart | No | Yes |

→ [`failures/failure_3_context_collapse/`](failures/failure_3_context_collapse/README.md)

---

### 4 · Schema Drift Bomb

**The symptom:** System worked perfectly for 3 weeks. On a Monday, every supplier query returned "Supplier not found." No code deployed. No config changed.

**Everyone assumed:** SAP sync failure. Stale Redis cache. Stale vectors. Two days of debugging before anyone checked the database schema.

**What actually happened:** The database team ran a quarterly normalization: `suppliers.supplier_id` → `suppliers.supplier_code`. SQLite silently returns empty results for queries against a renamed column. Nobody on the AI team was in the migration notification list.

**The fix:** Schema drift detection as a CI gate. A SHA-256 checksum of the schema is stored before any migration. After migration, a diff runs automatically and blocks deploy if breaking changes are detected.

```
⚠️  SCHEMA DRIFT DETECTED
🔴 COLUMN_DROPPED — suppliers.supplier_id
   Impact: All supplier lookups will return 0 rows silently
🟢 COLUMN_ADDED — suppliers.supplier_code
```

| | Before | After |
|--|--------|-------|
| Time to detect drift | 2 days (manual) | < 60 seconds |
| Deploy blocked on drift | No | Yes |

→ [`failures/failure_4_schema_drift_bomb/`](failures/failure_4_schema_drift_bomb/README.md) · [`src/data/schema_monitor.py`](src/data/schema_monitor.py)

---

### 5 · Auth Deadlock

**The symptom:** 40% of users got empty responses on supplier status queries. No errors in logs. The other 60% were fine. Non-reproducible in testing. Spent a week checking API uptime and rate limits — all normal.

**What actually happened:** OAuth2 token refresh race condition. When a token expired during a concurrent burst, multiple coroutines simultaneously detected expiry and hit the token endpoint. The endpoint used rotate-on-use security: each call issued a new token and invalidated the previous one. The first coroutine won. Every other got a token that was immediately invalid. Failures were silently swallowed as empty results.

**The fix:** `asyncio.Lock()` with double-checked locking. One coroutine refreshes; the rest wait and reuse.

```python
async def _get_valid_token(self) -> str:
    if self._token and not self._token.is_expired():
        return self._token.access_token        # Fast path — no lock needed

    async with self._refresh_lock:             # Only one refresher at a time
        if self._token and not self._token.is_expired():
            return self._token.access_token    # Already refreshed while we waited
        await self._refresh_token()

    return self._token.access_token
```

| Metric | Before | After |
|--------|--------|-------|
| Supplier status failure rate | ~40% | < 0.1% |
| Concurrent token refreshes | Up to N | Always 1 |
| Silent failures | Yes | No — explicit `SupplierAPIAuthError` |

→ [`failures/failure_5_auth_deadlock/`](failures/failure_5_auth_deadlock/README.md)

---

### 6 · The Eval That Lied

**The symptom:** Offline eval: 96% accuracy. Production week 2: 61%. Customer feedback: *"This thing is basically useless for our team."*

**Everyone assumed:** Model needs fine-tuning. RAG needs improvement. Raise the threshold.

**What actually happened:** The eval dataset was built by the ML team writing clean, well-formed English questions. Real procurement analysts write:

```
"wat r the penaltis for apex if they deliver late"      ← typos
"Quelles sont les pénalités d'Apex?"                    ← multilingual
"was the fine amount changed in the 2023 amendment?"    ← multi-hop
"what are apex's net-60 terms?"  [they have net-30]     ← wrong assumption
```

The eval was measuring the wrong distribution.

**The fix:** LLM-as-attacker generates 50 adversarial cases per CI run across 7 attack types. Adversarial pass rate ≥ 90% is now the deploy gate. Clean eval is a regression guard only.

| Phase | Clean acc. | Adversarial acc. | Production acc. |
|-------|-----------|-----------------|-----------------|
| Before | 96% | (not measured) | 61% |
| After | 94% | 83% | 85% |

Gap between offline and production: **35pp → 2pp.**

→ [`failures/failure_6_eval_lies/`](failures/failure_6_eval_lies/README.md) · [`src/evals/adversarial_gen.py`](src/evals/adversarial_gen.py)

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Agent | LangGraph | Stateful multi-agent graph with checkpointing |
| LLM | Claude Sonnet + Groq Llama 3.3 70B | Reasoning + live demo |
| RAG | Qdrant + BGE-M3 + BM25 + Cross-Encoder | Hybrid retrieval + reranking |
| Data | DuckDB + SQLite | OLAP + OLTP enterprise pattern |
| Cache | Redis | Semantic cache — 38% hit rate, 13.4s → 800ms |
| Observability | LangFuse + Prometheus + Grafana | Every failure was found via traces |
| Evals | DeepEval + LLM-as-judge | Faithfulness + adversarial CI gate |
| Auth | OAuth2 + asyncio.Lock | Mutex token refresh |
| CI/CD | GitHub Actions | Eval gate blocks deploy on regression |

---

## Skills Coverage

| FDE Requirement | Where it's demonstrated |
|-----------------|------------------------|
| Python, production code | All 6 fixes — async, typed, production-grade |
| Agent orchestration | LangGraph multi-agent with supervisor + specialists |
| RAG pipelines | Failure 1 — table-aware chunking, hybrid retrieval, cross-encoder reranking |
| Eval frameworks | Failure 6 — adversarial generation + LLM-as-judge CI gate |
| Observability | LangFuse + Prometheus — every failure diagnosed via traces |
| Redis / caching | Failure 2 — semantic cache, latency fix |
| OAuth2 / auth | Failure 5 — mutex refresh, explicit error surfacing |
| Schema / data engineering | Failure 4 — drift detection pipeline |
| Problem decomposition | 6 failure classes, each with wrong-diagnosis trap documented |

---

*"The measure of an engineer isn't the systems they build. It's the systems they walk into, diagnose under pressure, and leave better than they found them."*
