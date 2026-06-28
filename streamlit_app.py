"""
Meridian Manufacturing Corp — Procurement Intelligence Agent
============================================================
Interactive Streamlit demo for the AI Deployment Autopsy portfolio project.

Run locally:
    pip install streamlit anthropic
    streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io → New app → select this repo
    3. Add ANTHROPIC_API_KEY in Settings → Secrets
"""

from __future__ import annotations

import json
import time
from typing import Generator

import anthropic
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian Procurement Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Mock data ─────────────────────────────────────────────────────────────────
SUPPLIERS: dict[str, dict] = {
    "SUP-0001": {
        "name": "Apex Industries", "country": "CN", "risk_score": 0.82,
        "category": "Raw Materials", "annual_spend_usd": 28_400_000,
        "on_time_delivery_rate": 0.87, "quality_rejection_rate": 0.04,
        "open_disputes": 1, "status": "active",
        "certifications": ["ISO-9001", "ISO-14001"],
    },
    "SUP-0002": {
        "name": "Brightfield Components", "country": "DE", "risk_score": 0.21,
        "category": "Electronic Components", "annual_spend_usd": 54_200_000,
        "on_time_delivery_rate": 0.97, "quality_rejection_rate": 0.01,
        "open_disputes": 0, "status": "active",
        "certifications": ["ISO-9001", "IATF-16949", "AS9100"],
    },
    "SUP-0003": {
        "name": "Dalton Materials", "country": "MX", "risk_score": 0.67,
        "category": "Packaging", "annual_spend_usd": 9_100_000,
        "on_time_delivery_rate": 0.71, "quality_rejection_rate": 0.09,
        "open_disputes": 4, "status": "probation",
        "certifications": ["ISO-9001"],
    },
    "SUP-0004": {
        "name": "CoreTech Systems", "country": "US", "risk_score": 0.33,
        "category": "MRO", "annual_spend_usd": 18_700_000,
        "on_time_delivery_rate": 0.93, "quality_rejection_rate": 0.02,
        "open_disputes": 0, "status": "active",
        "certifications": ["ISO-9001", "ISO-27001"],
    },
    "SUP-0005": {
        "name": "Pinnacle Logistics", "country": "SG", "risk_score": 0.44,
        "category": "Logistics", "annual_spend_usd": 31_500_000,
        "on_time_delivery_rate": 0.89, "quality_rejection_rate": 0.00,
        "open_disputes": 2, "status": "active",
        "certifications": ["ISO-9001", "ISO-28000"],
    },
}

CONTRACTS = [
    {
        "contract_id": "CTR-00001", "supplier_id": "SUP-0001",
        "title": "Raw Materials Supply Agreement — FY2025",
        "value_usd": 28_400_000, "start_date": "2025-01-01", "end_date": "2025-12-31",
        "auto_renewal": True,
        "penalty_clauses": "1.0% per day late delivery, capped at 20% of PO value",
        "payment_terms": "Net-30",
    },
    {
        "contract_id": "CTR-00002", "supplier_id": "SUP-0002",
        "title": "Electronic Components Master Agreement",
        "value_usd": 54_200_000, "start_date": "2024-07-01", "end_date": "2026-06-30",
        "auto_renewal": False,
        "penalty_clauses": "0.5% per day late delivery, capped at 15% of PO value",
        "payment_terms": "Net-45",
    },
    {
        "contract_id": "CTR-00003", "supplier_id": "SUP-0003",
        "title": "Packaging Materials Framework Agreement",
        "value_usd": 9_100_000, "start_date": "2025-03-01", "end_date": "2026-02-28",
        "auto_renewal": True,
        "penalty_clauses": "2.0% per day late delivery, capped at 25% of PO value",
        "payment_terms": "Net-15",
    },
]

# ── Failure mode definitions ───────────────────────────────────────────────────
FAILURES = [
    {
        "id": 1, "emoji": "🔪",
        "title": "Hallucination Cascade",
        "subtitle": "Table-Blind RAG Chunking",
        "severity": "CRITICAL",
        "symptom": "Agent quoted penalty rates 2–5× the actual contract values. Three POs were flagged for incorrect supplier deductions worth ~$240K.",
        "root_cause": "The naive character chunker (chunk_size=800) split penalty tables mid-row. The LLM received fragments like '| Day 61 and' without context — then hallucinated the rest.",
        "fix": "TableAwareChunker: PyMuPDF block analysis detects pipe/tab tables and emits them as atomic chunks. Text-only blocks split at sentence boundaries.",
        "metric_before": "Hallucination rate: 23%",
        "metric_after": "Hallucination rate: 2%",
        "broken_code": '''\
class NaiveCharacterChunker:
    """BROKEN: Splits every 800 chars. Tables get cut mid-row."""
    def chunk(self, text: str, doc_id: str) -> list[dict]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append({
                "content": text[i : i + self.chunk_size],
                "chunk_type": "text",  # Everything labelled "text" — no table awareness
            })
        return chunks
''',
        "fixed_code": '''\
class TableAwareChunker:
    """FIXED: Tables are atomic. Text splits at sentence boundaries."""
    def chunk_text(self, text: str, doc_id: str) -> list[DocumentChunk]:
        blocks = self._extract_blocks(text)
        chunks = []
        for block in blocks:
            chunk_type = self._classify_text_block(block)  # TABLE or TEXT
            if chunk_type == ChunkType.TABLE:
                chunks.append(DocumentChunk(          # Always atomic
                    content=block, chunk_type=ChunkType.TABLE, ...
                ))
            else:
                chunks.extend(self._split_at_sentences(block, ...))
        return chunks
''',
    },
    {
        "id": 2, "emoji": "🐢",
        "title": "Latency Wall",
        "subtitle": "Sequential Tool Execution",
        "severity": "HIGH",
        "symptom": "P95 query latency: 47 seconds. 93% user abandonment on multi-supplier queries. The agent timed out before completing approval chains.",
        "root_cause": "Tools ran serially: search_contracts (8s) → lookup_supplier × 5 (6s each) → flag_risks (4s). Minimum 42s for a 5-supplier query.",
        "fix": "asyncio.gather() fans out all supplier lookups in parallel. Redis semantic cache (cosine sim > 0.92) returns cached answers in 2ms instead of 8s.",
        "metric_before": "P95 latency: 47s",
        "metric_after": "P95 latency: 4.2s",
        "broken_code": '''\
# BROKEN: Sequential — each lookup blocks the next
results = []
for supplier_id in supplier_ids:
    supplier = await lookup_supplier(supplier_id)  # 6s each
    results.append(supplier)
# Total for 5 suppliers: 30s just on lookups
''',
        "fixed_code": '''\
# FIXED: All lookups fire simultaneously
results = await asyncio.gather(
    *[lookup_supplier(sid) for sid in supplier_ids],
    return_exceptions=True,
)
# Total: max(individual latencies) ≈ 6s regardless of N suppliers

# Plus semantic cache for repeat queries
cached = await cache.get(query, namespace="contracts")
if cached:
    return cached  # 2ms, not 8s
''',
    },
    {
        "id": 3, "emoji": "🧠",
        "title": "Context Collapse",
        "subtitle": "Unbounded Conversation State",
        "severity": "HIGH",
        "symptom": "After ~12 turns, the agent forgot supplier IDs from turn 2, contradicted earlier risk assessments, and duplicated approval submissions it had already made.",
        "root_cause": "LangGraph accumulated every message with no pruning. By turn 12, the context window was 98% full. The LLM silently truncated early messages.",
        "fix": "Every 8 turns, an LLM compression pass summarizes old messages into a single SystemMessage — explicitly preserving all IDs and approval state.",
        "metric_before": "Context overflow at turn 12",
        "metric_after": "Stable across 100+ turns",
        "broken_code": '''\
# BROKEN: Messages grow forever
class ProcurementState(TypedDict):
    messages: Annotated[list, add_messages]
    # By turn 12: ~45,000 tokens. LLM silently drops early messages.
    # Approval IDs from turn 3 → gone. Agent re-submits approvals.
''',
        "fixed_code": '''\
# FIXED: Compress every 8 turns, preserve critical state
async def summarize_if_needed(state: ProcurementState):
    if state["turn_count"] % 8 != 0:
        return state
    summary = await llm.ainvoke([
        SystemMessage("Summarize. Preserve ALL supplier IDs, "
                      "approval IDs, and risk flags verbatim."),
        *state["messages"][:-4],
    ])
    state["messages"] = [
        SystemMessage(f"[SUMMARY] {summary.content}"),
        *state["messages"][-4:],  # Keep last 4 turns verbatim
    ]
    return state
''',
    },
    {
        "id": 4, "emoji": "💣",
        "title": "Schema Drift Bomb",
        "subtitle": "Silent Database Migration",
        "severity": "CRITICAL",
        "symptom": "'Supplier not found' for every query — 72 hours after a DB migration. No errors logged. System appeared healthy in all monitors.",
        "root_cause": "A migration renamed supplier_id → supplier_code. Application code still queried WHERE supplier_id = ?. SQLite returned 0 rows silently — no exception.",
        "fix": "Schema snapshot → diff → CI gate. Any CRITICAL change (column drop/rename) exits 1 and blocks the deployment pipeline.",
        "metric_before": "72h MTTR, 0 automated alerts",
        "metric_after": "Caught at CI, 0 production impact",
        "broken_code": '''\
# BROKEN: Migration ran silently, nobody noticed the rename
-- migration_007.sql
ALTER TABLE suppliers RENAME COLUMN supplier_id TO supplier_code;

# Application code still uses old column name
async def get_supplier(supplier_id: str):
    cursor = await db.execute(
        "SELECT * FROM suppliers WHERE supplier_id = ?",  # 0 rows returned
        (supplier_id,)
    )
    # Returns None silently — no error, no log, no alert
''',
        "fixed_code": '''\
# FIXED: CI gate blocks deploy on drift
$ python -m src.data.schema_monitor diff

[CRITICAL] COLUMN_DROPPED: suppliers.supplier_id
  Impact: 7 queries in 3 modules reference this column.
  Breaking: get_supplier(), search_suppliers(), get_procurement_analytics()

Drift detected. Exiting with code 1.
# → GitHub Actions step fails → PR cannot merge ✓

# .github/workflows/eval-gate.yml
- name: Schema drift check
  run: python -m src.data.schema_monitor diff
  # exits 1 on CRITICAL drift → deployment blocked
''',
    },
    {
        "id": 5, "emoji": "🔒",
        "title": "Auth Deadlock",
        "subtitle": "OAuth2 Token Race Condition",
        "severity": "HIGH",
        "symptom": "Under load, system froze for 30–90s every hour. Affected all concurrent users simultaneously. Recovery was spontaneous and unexplained.",
        "root_cause": "10 coroutines detected the expired token simultaneously and all called _refresh_token() concurrently. The token endpoint 429'd 9 of them, causing cascading backoff storms.",
        "fix": "asyncio.Lock() with double-checked locking. One coroutine refreshes; the rest wait, then take the fast path when the lock is released.",
        "metric_before": "P99 latency: 90s at token expiry",
        "metric_after": "P99 latency: 4.8s (single refresh)",
        "broken_code": '''\
# BROKEN: No lock — all coroutines refresh simultaneously
class BrokenSupplierAPIClient:
    async def _get_valid_token(self) -> str:
        if not self._token or self._token.is_expired():
            await self._refresh_token()  # All 10 callers hit this
        return self._token.access_token  # Token endpoint 429s → storm
''',
        "fixed_code": '''\
# FIXED: Mutex + double-checked locking
class SupplierAPIClient:
    def __init__(self):
        self._refresh_lock = asyncio.Lock()  # THE KEY FIX

    async def _get_valid_token(self) -> str:
        # Fast path: skip lock entirely if token valid
        if self._token and not self._token.is_expired():
            return self._token.access_token

        async with self._refresh_lock:
            # Double-check: coroutine B sees fresh token from A
            if self._token and not self._token.is_expired():
                return self._token.access_token  # No refresh needed
            await self._refresh_token()  # Only ONE coroutine reaches this
        return self._token.access_token
''',
    },
    {
        "id": 6, "emoji": "📊",
        "title": "Eval Lies",
        "subtitle": "Benchmark Overfitting",
        "severity": "HIGH",
        "symptom": "Eval suite reported 94% pass rate. Production showed 61% user satisfaction. Gap widened each sprint as team 'improved' eval scores.",
        "root_cause": "The eval set was built from the same distribution as training prompts. No typos, multilingual queries, multi-hop reasoning, or wrong-assumption queries — exactly what real users send.",
        "fix": "LLM-as-attacker generates 50 adversarial cases per run (7 attack types). CI gate requires 90% pass rate on adversarial set, not just the clean set.",
        "metric_before": "Clean eval: 94% | Production: 61%",
        "metric_after": "Adversarial eval: 91% | Production: 89%",
        "broken_code": '''\
# BROKEN: Only clean, perfectly-phrased English queries
EVAL_CASES = [
    {"query": "What are the penalty clauses for Apex Industries?"},
    {"query": "Show suppliers with risk score above 0.7"},
    {"query": "When does contract CTR-00001 expire?"},
    # Never tests what real users actually send
]
''',
        "fixed_code": '''\
# FIXED: LLM-as-attacker generates adversarial cases
cases = await generator.generate(n=50, attack_types=[
    "typo",             # "penalti clouses for apex"
    "multilingual",     # "¿Cuáles son las penalizaciones?"
    "multi_hop",        # "Which suppliers from same country as
                        #  highest-risk vendor have expiring contracts?"
    "wrong_assumption", # "What's Apex's 99% on-time rate?" (it's 87%)
    "ambiguous",        # "check the contract" (no supplier specified)
    "informal",         # "yo what's the deal with dalton"
    "truncated",        # "penalty for late del"
])
assert report.adversarial_pass_rate >= 0.90  # CI gate
''',
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are the Procurement Intelligence Agent for Meridian Manufacturing Corp, \
a $2.4B industrial manufacturer with 1,200+ active supplier contracts.

You help procurement managers with supplier risk assessment, contract terms lookup, \
purchase order approval routing, spend analytics, and compliance flagging.

LIVE SUPPLIER DATABASE:
{json.dumps(SUPPLIERS, indent=2)}

ACTIVE CONTRACTS:
{json.dumps(CONTRACTS, indent=2)}

RISK & APPROVAL RULES:
- Risk score > 0.70 → HIGH RISK → VP + CPO approval required for POs > $500K
- Risk score 0.40–0.70 → MEDIUM → standard approval chain
- Risk score < 0.40 → LOW RISK → auto-approve up to $2M
- PO < $100K: Procurement Manager only
- PO $100K–$1M: VP Procurement
- PO > $1M: CFO + VP Procurement

Always cite contract IDs when referencing contract terms. Flag risks prominently. \
Be concise and professional. Format all amounts as USD."""

# ── Helper: risk color ────────────────────────────────────────────────────────
def risk_color(score: float) -> str:
    if score >= 0.70:
        return "🔴"
    if score >= 0.40:
        return "🟡"
    return "🟢"


def risk_label(score: float) -> str:
    if score >= 0.70:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.shields.io/badge/Meridian%20Manufacturing-Procurement%20AI-4a9eff?style=for-the-badge",
        use_container_width=True,
    )
    st.markdown("---")

    # API key
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get yours at console.anthropic.com",
        value=st.secrets.get("ANTHROPIC_API_KEY", ""),
    )

    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigate",
        ["🤖 Agent Chat", "💥 Failure Autopsies", "📐 Architecture"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Supplier quick-reference
    st.markdown("**Supplier Quick Reference**")
    for sid, s in SUPPLIERS.items():
        col1, col2 = st.columns([3, 1])
        col1.caption(f"{risk_color(s['risk_score'])} {s['name']}")
        col2.caption(risk_label(s["risk_score"]))

    st.markdown("---")
    st.caption("AI Deployment Autopsy Portfolio · [GitHub](https://github.com/janhavijadhav/ai-deployment-autopsy)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENT CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🤖 Agent Chat":
    st.title("🏭 Procurement Intelligence Agent")
    st.caption("Meridian Manufacturing Corp · $2.4B spend · 1,200+ supplier contracts")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Suppliers", "1,247", "+12 this quarter")
    col2.metric("Annual Spend", "$2.4B", "+8.3% YoY")
    col3.metric("High Risk Suppliers", "3", "↓1 from last month")
    col4.metric("Contracts Expiring (90d)", "47", "⚠️ 12 auto-renew")

    st.markdown("---")

    # Suggested prompts
    with st.expander("💡 Example queries to try", expanded=False):
        examples = [
            "What are the penalty clauses for Apex Industries?",
            "Which suppliers have risk scores above 0.7 and open disputes?",
            "I need to approve a $750K PO for Dalton Materials — what's the approval chain?",
            "Compare on-time delivery rates across all 5 suppliers.",
            "When does the Brightfield contract expire and does it auto-renew?",
            "Flag all high-risk suppliers and summarize their contract exposure.",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state.setdefault("messages", [])
                st.session_state["messages"].append({"role": "user", "content": ex})
                st.rerun()

    # Init chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🏭" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about suppliers, contracts, risks, or approvals…"):
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Stream response from Claude
        with st.chat_message("assistant", avatar="🏭"):
            client = anthropic.Anthropic(api_key=api_key)
            message_placeholder = st.empty()
            full_response = ""

            start = time.time()
            try:
                with client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state["messages"]
                    ],
                ) as stream:
                    for text in stream.text_stream:
                        full_response += text
                        message_placeholder.markdown(full_response + "▌")
                latency = (time.time() - start) * 1000
                message_placeholder.markdown(full_response)
                st.caption(f"⚡ {latency:.0f}ms · claude-sonnet-4-6 · {len(full_response)} chars")
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Check your key at console.anthropic.com.")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.session_state["messages"].append({"role": "assistant", "content": full_response})

    # Clear chat
    if st.session_state.get("messages"):
        if st.button("🗑️ Clear chat", type="secondary"):
            st.session_state["messages"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAILURE AUTOPSIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💥 Failure Autopsies":
    st.title("💥 Production Failure Autopsies")
    st.markdown(
        "Six real failure classes diagnosed and fixed in a multi-agent procurement platform. "
        "Each card shows the symptom, root cause, fix, and before/after code."
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Failures Documented", "6", "All fixed")
    col2.metric("MTTR Improvement", "72h → CI gate", "Failure 4")
    col3.metric("Latency Improvement", "47s → 4.2s", "Failure 2")

    st.markdown("---")

    severity_color = {"CRITICAL": "🔴", "HIGH": "🟠"}

    for f in FAILURES:
        with st.expander(
            f"{f['emoji']} Failure {f['id']}: {f['title']} — {f['subtitle']}  "
            f"{severity_color.get(f['severity'], '')} {f['severity']}",
            expanded=False,
        ):
            col_l, col_r = st.columns(2)

            with col_l:
                st.markdown("**Symptom**")
                st.info(f["symptom"])
                st.markdown("**Root Cause**")
                st.warning(f["root_cause"])
                st.markdown("**Fix**")
                st.success(f["fix"])

                # Metrics
                c1, c2 = st.columns(2)
                c1.metric("Before", f["metric_before"])
                c2.metric("After", f["metric_after"])

            with col_r:
                st.markdown("**❌ Broken Code**")
                st.code(f["broken_code"], language="python")
                st.markdown("**✅ Fixed Code**")
                st.code(f["fixed_code"], language="python")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Architecture":
    st.title("📐 System Architecture")
    st.caption("Multi-agent procurement intelligence platform · Meridian Manufacturing Corp")

    st.markdown("---")

    # Architecture diagram (ASCII → looks great in code block)
    st.markdown("### Agent Pipeline")
    st.code("""\
User Query
    │
    ▼
FastAPI /query endpoint
    │
    ▼
LangGraph Stateful Agent
    ├── route_query (intent classification)
    ├── summarize_if_needed (Failure 3 fix — every 8 turns)
    ├── agent node (Claude claude-sonnet-4-6)
    └── tools node
         ├── search_contracts ──→ Hybrid RAG
         │                         ├── Dense: BGE-M3 → Qdrant
         │                         ├── Sparse: BM25Okapi
         │                         └── RRF Fusion + Table Boost (Failure 1 fix)
         ├── lookup_supplier ───→ SQLite (async)
         ├── get_live_status ───→ Supplier REST API (OAuth2 mutex — Failure 5 fix)
         ├── flag_risks ────────→ asyncio.gather parallel (Failure 2 fix)
         └── initiate_approval → Threshold-based chain routing

State: SQLite checkpointing (LangGraph AsyncSqliteSaver)
Cache: Redis semantic cache cosine > 0.92 (Failure 2 fix)
Schema guard: snapshot → diff → CI exit 1 (Failure 4 fix)
Evals: LLM-as-judge + LLM-as-attacker adversarial (Failure 6 fix)
""", language="text")

    st.markdown("### Tech Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🤖 Agent Layer**")
        stack_agent = {
            "Orchestration": "LangGraph",
            "LLM": "Claude claude-sonnet-4-6 (Anthropic)",
            "State": "SQLite checkpointing",
            "Tools": "7 async @tool functions",
        }
        for k, v in stack_agent.items():
            st.markdown(f"- **{k}**: {v}")

        st.markdown("**🔍 RAG Layer**")
        stack_rag = {
            "Dense retrieval": "BGE-M3 embeddings → Qdrant",
            "Sparse retrieval": "BM25Okapi",
            "Fusion": "Reciprocal Rank Fusion (k=60)",
            "Chunking": "TableAwareChunker (PyMuPDF)",
        }
        for k, v in stack_rag.items():
            st.markdown(f"- **{k}**: {v}")

    with col2:
        st.markdown("**🗄️ Data Layer**")
        stack_data = {
            "OLTP": "SQLite (aiosqlite)",
            "OLAP": "DuckDB",
            "Cache": "Redis semantic cache",
            "Schema guard": "Snapshot diff + CI gate",
        }
        for k, v in stack_data.items():
            st.markdown(f"- **{k}**: {v}")

        st.markdown("**🔐 Auth**")
        stack_auth = {
            "Protocol": "OAuth2 client_credentials",
            "Token safety": "asyncio.Lock double-checked",
            "Retry": "Exponential backoff (3 retries)",
            "Error": "Explicit SupplierAPIError (not silent None)",
        }
        for k, v in stack_auth.items():
            st.markdown(f"- **{k}**: {v}")

    with col3:
        st.markdown("**📊 Observability**")
        stack_obs = {
            "Tracing": "LangFuse (self-hosted)",
            "Metrics": "Prometheus + Grafana",
            "Alerts": "4 alert rules (latency, drift, auth, eval)",
            "API": "FastAPI + /health + /ready",
        }
        for k, v in stack_obs.items():
            st.markdown(f"- **{k}**: {v}")

        st.markdown("**🧪 Eval & CI/CD**")
        stack_eval = {
            "Judge": "LLM-as-judge (faithfulness + relevance)",
            "Adversarial": "LLM-as-attacker (7 attack types)",
            "Framework": "DeepEval",
            "CI": "GitHub Actions 5-stage pipeline",
        }
        for k, v in stack_eval.items():
            st.markdown(f"- **{k}**: {v}")

    st.markdown("---")
    st.markdown("### CI/CD Pipeline (GitHub Actions)")
    st.code("""\
Stage 1: schema-check          python -m src.data.schema_monitor diff
    ↓ (exit 0 only)
Stage 2: eval-clean            pytest tests/ (unit tests — 39 tests)
    ↓
Stage 3: eval-adversarial      python -m src.evals.eval_runner --mode adversarial
    ↓ (pass rate ≥ 90%)
Stage 4: eval-faithfulness     python -m src.evals.eval_runner --mode faithfulness
    ↓ (faithfulness ≥ 85%, relevance ≥ 80%)
Stage 5: build-and-push        docker build + push to registry
""", language="text")

    st.markdown("### Supplier Risk Overview")
    supplier_rows = [
        {
            "Supplier": s["name"],
            "Country": s["country"],
            "Risk Score": s["risk_score"],
            "Risk Level": risk_label(s["risk_score"]),
            "On-Time %": f"{s['on_time_delivery_rate']*100:.0f}%",
            "Annual Spend": f"${s['annual_spend_usd']:,.0f}",
            "Status": s["status"].upper(),
        }
        for s in SUPPLIERS.values()
    ]
    st.dataframe(supplier_rows, use_container_width=True, hide_index=True)
