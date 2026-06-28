"""
Meridian Manufacturing Corp — Procurement Intelligence Agent
============================================================
Interactive Streamlit demo for the AI Deployment Autopsy portfolio project.
Uses Groq's free API (Llama 3.3 70B) — no credit card required.

Run locally:
    pip install streamlit groq
    streamlit run streamlit_app.py

Deploy: share.streamlit.io  |  Secrets: GROQ_API_KEY = "gsk_..."
Free key at: console.groq.com
"""

from __future__ import annotations
import json
import time
from groq import Groq, AuthenticationError as GroqAuthError
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meridian Procurement Intelligence",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default streamlit header */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Typography */
h1 { font-size: 1.8rem !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; }

/* Metric before/after boxes */
.metric-box {
    border-radius: 8px;
    padding: 14px 18px;
    margin: 4px 0;
}
.metric-before {
    background: rgba(255, 75, 75, 0.12);
    border: 1px solid rgba(255, 75, 75, 0.4);
}
.metric-after {
    background: rgba(33, 195, 84, 0.12);
    border: 1px solid rgba(33, 195, 84, 0.4);
}
.metric-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    opacity: 0.7;
    margin-bottom: 4px;
}
.metric-value { font-size: 14px; font-weight: 500; }
.metric-before .metric-label { color: #ff4b4b; }
.metric-after  .metric-label { color: #21c354; }

/* Severity badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
}
.badge-critical { background: rgba(255,75,75,0.15); color: #ff4b4b; border: 1px solid rgba(255,75,75,0.4); }
.badge-high     { background: rgba(255,165,0,0.15);  color: #ffa500; border: 1px solid rgba(255,165,0,0.4); }

/* Failure card number */
.failure-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px; height: 32px;
    border-radius: 50%;
    background: #4a9eff22;
    border: 1px solid #4a9eff66;
    color: #4a9eff;
    font-weight: 700;
    font-size: 14px;
    margin-right: 10px;
}

/* Sidebar supplier risk dots */
.sup-row { display: flex; justify-content: space-between; align-items: center; margin: 2px 0; font-size: 12px; }
.dot-high   { color: #ff4b4b; }
.dot-medium { color: #ffa500; }
.dot-low    { color: #21c354; }

/* Architecture pipeline node */
.pipe-node {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 8px;
    padding: 10px 16px;
    text-align: center;
    font-size: 13px;
    font-weight: 500;
}
.pipe-node-primary { border-color: #4a9eff66; background: #4a9eff11; }
.pipe-arrow { text-align: center; color: #4a9eff; font-size: 20px; line-height: 1; margin: 2px 0; }

/* Stack card */
.stack-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}
.stack-card h4 {
    margin: 0 0 10px 0;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #4a9eff;
}
.stack-row {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 0;
    border-bottom: 1px solid #2d325044;
}
.stack-row:last-child { border-bottom: none; }
.stack-key { opacity: 0.6; }
.stack-val { font-weight: 500; text-align: right; }
</style>
""", unsafe_allow_html=True)

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

# ── Failure data ──────────────────────────────────────────────────────────────
FAILURES = [
    {
        "id": 1, "title": "Hallucination Cascade", "subtitle": "Table-Blind RAG Chunking",
        "severity": "CRITICAL",
        "symptom": "Agent quoted penalty rates 2–5× the actual contract values. Three POs flagged for incorrect supplier deductions totalling ~$240K.",
        "root_cause": "The naive character chunker (chunk_size=800) split penalty tables mid-row. The LLM received fragments like '| Day 61 and' without context — then hallucinated the missing values.",
        "fix": "TableAwareChunker detects pipe/tab tables via PyMuPDF block analysis and emits them as atomic chunks. Text-only blocks split at sentence boundaries only.",
        "metric_before": "Hallucination rate: 23%",
        "metric_after": "Hallucination rate: 2%",
        "broken_code": """\
class NaiveCharacterChunker:
    \"\"\"BROKEN: Splits every 800 chars. Tables get cut mid-row.\"\"\"
    def chunk(self, text: str, doc_id: str) -> list[dict]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append({
                "content": text[i : i + self.chunk_size],
                "chunk_type": "text",  # Everything labelled "text" — no table awareness
            })
        return chunks""",
        "fixed_code": """\
class TableAwareChunker:
    \"\"\"FIXED: Tables are atomic. Text splits at sentence boundaries.\"\"\"
    def chunk_text(self, text: str, doc_id: str) -> list[DocumentChunk]:
        blocks = self._extract_blocks(text)
        chunks = []
        for block in blocks:
            chunk_type = self._classify_text_block(block)  # TABLE or TEXT
            if chunk_type == ChunkType.TABLE:
                chunks.append(DocumentChunk(    # Always atomic — never split
                    content=block,
                    chunk_type=ChunkType.TABLE,
                    ...
                ))
            else:
                chunks.extend(self._split_at_sentences(block, ...))
        return chunks""",
    },
    {
        "id": 2, "title": "Latency Wall", "subtitle": "Sequential Tool Execution",
        "severity": "HIGH",
        "symptom": "P95 query latency hit 47 seconds. 93% user abandonment on multi-supplier queries. Agent timed out before completing approval chains.",
        "root_cause": "Tools ran serially: search_contracts (8s) → lookup_supplier × 5 (6s each) → flag_risks (4s). Minimum 42s for any query involving 5+ suppliers.",
        "fix": "asyncio.gather() fans out all supplier lookups simultaneously. Redis semantic cache (cosine sim > 0.92) returns cached answers in 2ms instead of 8s.",
        "metric_before": "P95 latency: 47 seconds",
        "metric_after": "P95 latency: 4.2 seconds",
        "broken_code": """\
# BROKEN: Sequential — each lookup blocks the next
results = []
for supplier_id in supplier_ids:
    supplier = await lookup_supplier(supplier_id)  # 6s each, blocking
    results.append(supplier)
# Total for 5 suppliers: 30s on lookups alone""",
        "fixed_code": """\
# FIXED: All lookups fire simultaneously
results = await asyncio.gather(
    *[lookup_supplier(sid) for sid in supplier_ids],
    return_exceptions=True,
)
# Total: max(individual latencies) ≈ 6s regardless of N suppliers

# Semantic cache for repeat queries
cached = await cache.get(query, namespace="contracts")
if cached:
    return cached  # 2ms, not 8s""",
    },
    {
        "id": 3, "title": "Context Collapse", "subtitle": "Unbounded Conversation State",
        "severity": "HIGH",
        "symptom": "After 12 turns, the agent forgot supplier IDs agreed in turn 2, contradicted earlier risk assessments, and submitted duplicate approval requests.",
        "root_cause": "LangGraph accumulated every message with no pruning. By turn 12, the context was 98% full. The LLM silently truncated early messages — including approval IDs.",
        "fix": "Every 8 turns, an LLM compression pass summarizes old messages into a single SystemMessage, explicitly preserving all IDs, approval chains, and risk flags.",
        "metric_before": "Context overflow at turn 12",
        "metric_after": "Stable across 100+ turns",
        "broken_code": """\
# BROKEN: Messages grow forever — no pruning
class ProcurementState(TypedDict):
    messages: Annotated[list, add_messages]
    # By turn 12: ~45,000 tokens
    # LLM silently drops early messages
    # Approval IDs from turn 3 → lost
    # Agent re-submits the same approvals""",
        "fixed_code": """\
# FIXED: Compress every 8 turns, preserve critical state
async def summarize_if_needed(state: ProcurementState):
    if state["turn_count"] % 8 != 0:
        return state
    summary = await llm.ainvoke([
        SystemMessage(
            "Summarize this conversation. Preserve ALL supplier IDs, "
            "approval IDs, and risk flags verbatim."
        ),
        *state["messages"][:-4],
    ])
    state["messages"] = [
        SystemMessage(f"[SUMMARY] {summary.content}"),
        *state["messages"][-4:],  # Always keep last 4 turns verbatim
    ]
    return state""",
    },
    {
        "id": 4, "title": "Schema Drift Bomb", "subtitle": "Silent Database Migration",
        "severity": "CRITICAL",
        "symptom": "'Supplier not found' for every query — 72 hours after a scheduled DB migration. No errors logged. All health monitors reported green.",
        "root_cause": "A migration renamed supplier_id → supplier_code. Application code still queried WHERE supplier_id = ?. SQLite returned 0 rows silently with no exception.",
        "fix": "Schema snapshot → diff → CI gate. Any CRITICAL change (column drop or rename) causes the pipeline to exit 1 and block deployment before it reaches production.",
        "metric_before": "72h MTTR, zero automated alerts",
        "metric_after": "Caught at CI, zero production impact",
        "broken_code": """\
-- BROKEN: Migration ran silently with no impact analysis
ALTER TABLE suppliers RENAME COLUMN supplier_id TO supplier_code;

# Application code still uses the old column name
async def get_supplier(supplier_id: str):
    cursor = await db.execute(
        "SELECT * FROM suppliers WHERE supplier_id = ?",
        (supplier_id,)
    )
    # Returns None silently — no error, no log, no alert fired""",
        "fixed_code": """\
# FIXED: CI gate detects drift before deploy
$ python -m src.data.schema_monitor diff

[CRITICAL] COLUMN_DROPPED: suppliers.supplier_id
  Impact: 7 queries in 3 modules reference this column.
  Breaking: get_supplier(), search_suppliers(), get_procurement_analytics()

Drift detected. Exiting with code 1.
# → GitHub Actions step fails → PR cannot merge

# .github/workflows/eval-gate.yml
- name: Schema drift check
  run: python -m src.data.schema_monitor diff
  # exits 1 on CRITICAL → deployment blocked""",
    },
    {
        "id": 5, "title": "Auth Deadlock", "subtitle": "OAuth2 Token Race Condition",
        "severity": "HIGH",
        "symptom": "Under load, the system froze for 30–90 seconds every hour, affecting all concurrent users simultaneously. Recovery was spontaneous with no clear cause.",
        "root_cause": "10 coroutines detected the expired token simultaneously and all called _refresh_token() concurrently. The token endpoint 429'd 9 of them, causing cascading backoff storms.",
        "fix": "asyncio.Lock() with double-checked locking. One coroutine refreshes; the rest wait, then take the fast path when the lock is released — one network call instead of ten.",
        "metric_before": "P99 latency: 90s at token expiry",
        "metric_after": "P99 latency: 4.8s (single refresh)",
        "broken_code": """\
# BROKEN: No lock — all coroutines refresh simultaneously
class BrokenSupplierAPIClient:
    async def _get_valid_token(self) -> str:
        if not self._token or self._token.is_expired():
            # All 10 concurrent callers enter here at the same time
            await self._refresh_token()
        return self._token.access_token
        # Token endpoint rate-limits at 3 req/s → 429 storm""",
        "fixed_code": """\
# FIXED: Mutex + double-checked locking
class SupplierAPIClient:
    def __init__(self):
        self._refresh_lock = asyncio.Lock()  # THE KEY FIX

    async def _get_valid_token(self) -> str:
        # Fast path: skip lock entirely if token is still valid
        if self._token and not self._token.is_expired():
            return self._token.access_token

        async with self._refresh_lock:
            # Double-check inside lock: coroutine B sees token already
            # refreshed by coroutine A and skips the refresh
            if self._token and not self._token.is_expired():
                return self._token.access_token
            await self._refresh_token()  # Only ONE coroutine reaches here
        return self._token.access_token""",
    },
    {
        "id": 6, "title": "Eval Lies", "subtitle": "Benchmark Overfitting",
        "severity": "HIGH",
        "symptom": "Eval suite showed 94% pass rate. Production showed 61% user satisfaction. The gap widened each sprint as the team optimised for the benchmark.",
        "root_cause": "The eval set was drawn from the same distribution as training prompts — clean, unambiguous, English-only. Real users sent typos, multilingual queries, and wrong assumptions.",
        "fix": "LLM-as-attacker generates 50 adversarial cases per run across 7 attack types. CI gate requires 90% pass rate on the adversarial set, not just the clean set.",
        "metric_before": "Clean: 94% pass  |  Production: 61%",
        "metric_after": "Adversarial: 91%  |  Production: 89%",
        "broken_code": """\
# BROKEN: Only clean, perfectly-phrased English queries
EVAL_CASES = [
    {"query": "What are the penalty clauses for Apex Industries?"},
    {"query": "Show suppliers with risk score above 0.7"},
    {"query": "When does contract CTR-00001 expire?"},
]
# Never tests what real users actually send""",
        "fixed_code": """\
# FIXED: LLM-as-attacker generates adversarial test cases
cases = await generator.generate(n=50, attack_types=[
    "typo",             # "penalti clouses for apex"
    "multilingual",     # "¿Cuáles son las penalizaciones?"
    "multi_hop",        # "Which suppliers from the same country as
                        #  our highest-risk vendor have expiring contracts?"
    "wrong_assumption", # "What's Apex's 99% on-time rate?" (it's 87%)
    "ambiguous",        # "check the contract" (no supplier named)
    "informal",         # "yo what's the deal with dalton"
    "truncated",        # "penalty for late del"
])
assert report.adversarial_pass_rate >= 0.90  # CI gate""",
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
- Risk score 0.40–0.70 → MEDIUM RISK → standard approval chain
- Risk score < 0.40 → LOW RISK → auto-approve up to $2M
- PO < $100K: Procurement Manager only
- PO $100K–$1M: VP Procurement
- PO > $1M: CFO + VP Procurement

Always cite contract IDs when referencing terms. Flag risks prominently. \
Be concise and professional. Format all amounts in USD."""

# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_dot(score: float) -> str:
    if score >= 0.70: return '<span class="dot-high">●</span>'
    if score >= 0.40: return '<span class="dot-medium">●</span>'
    return '<span class="dot-low">●</span>'

def risk_label(score: float) -> str:
    if score >= 0.70: return "HIGH"
    if score >= 0.40: return "MED"
    return "LOW"

def metric_boxes(before: str, after: str) -> str:
    return f"""
<div style="display:flex; gap:12px; margin:12px 0;">
  <div class="metric-box metric-before" style="flex:1">
    <div class="metric-label">Before</div>
    <div class="metric-value">{before}</div>
  </div>
  <div class="metric-box metric-after" style="flex:1">
    <div class="metric-label">After fix</div>
    <div class="metric-value">{after}</div>
  </div>
</div>"""

def call_llm(api_key: str, messages: list[dict]) -> str:
    """Blocking Groq call, returns full response string."""
    client = Groq(api_key=api_key)
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        stream=True,
    )
    result = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            result += delta
    return result

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
      <div style="font-size: 11px; font-weight: 700; letter-spacing: 2px;
                  text-transform: uppercase; color: #4a9eff; margin-bottom: 4px;">
        MERIDIAN MANUFACTURING
      </div>
      <div style="font-size: 16px; font-weight: 700; line-height: 1.2;">
        Procurement Intelligence
      </div>
      <div style="font-size: 11px; opacity: 0.5; margin-top: 2px;">
        $2.4B spend · 1,247 suppliers
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="gsk_...",
        value=st.secrets.get("GROQ_API_KEY", ""),
        label_visibility="visible",
    )
    st.caption("Free key → [console.groq.com](https://console.groq.com)")

    st.divider()

    page = st.radio(
        "Navigation",
        ["Agent Chat", "Failure Autopsies", "Architecture"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("<div style='font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; opacity:0.5; margin-bottom:8px;'>SUPPLIER WATCHLIST</div>", unsafe_allow_html=True)
    for s in SUPPLIERS.values():
        st.markdown(
            f'<div class="sup-row">'
            f'<span>{risk_dot(s["risk_score"])} {s["name"]}</span>'
            f'<span style="font-size:10px; opacity:0.5;">{risk_label(s["risk_score"])}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("AI Deployment Autopsy · [GitHub](https://github.com/janhavijadhav/ai-deployment-autopsy)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENT CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "Agent Chat":
    st.markdown("""
    <div style="margin-bottom: 8px;">
      <div style="font-size: 11px; font-weight: 700; letter-spacing: 2px;
                  text-transform: uppercase; color: #4a9eff;">MERIDIAN MANUFACTURING</div>
      <h1 style="margin: 4px 0 2px 0;">Procurement Intelligence Agent</h1>
      <div style="opacity: 0.5; font-size: 13px;">
        Multi-agent · Hybrid RAG · 1,200+ supplier contracts · LangGraph + Claude
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Suppliers", "1,247", "+12 this quarter")
    c2.metric("Annual Spend", "$2.4B", "+8.3% YoY")
    c3.metric("High Risk", "3 suppliers", "↓1 from last month")
    c4.metric("Expiring (90d)", "47 contracts", "12 auto-renew")

    st.divider()

    # Example queries
    with st.expander("Example queries", expanded=False):
        examples = [
            "What are the penalty clauses for Apex Industries?",
            "Which suppliers have risk scores above 0.7 and open disputes?",
            "I need to approve a $750K PO for Dalton Materials — what's the approval chain?",
            "Compare on-time delivery rates across all 5 suppliers.",
            "When does the Brightfield contract expire and does it auto-renew?",
            "Flag all high-risk suppliers and summarise their contract exposure.",
        ]
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.setdefault("messages", [])
                st.session_state["messages"].append({"role": "user", "content": ex})
                st.session_state["pending_query"] = ex
                st.rerun()

    # Init chat
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Render history
    for msg in st.session_state["messages"]:
        avatar = "🏭" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Determine active prompt (typed OR from button click)
    active_prompt: str | None = None
    add_to_history = True

    typed = st.chat_input("Ask about suppliers, contracts, risks, or approvals…")
    if typed:
        active_prompt = typed
    elif st.session_state.get("pending_query"):
        active_prompt = st.session_state.pop("pending_query")
        add_to_history = False  # Already added when button was clicked

    if active_prompt:
        if not api_key:
            st.error("Enter your Groq API key in the sidebar. Free at console.groq.com")
            st.stop()

        if add_to_history:
            st.session_state["messages"].append({"role": "user", "content": active_prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(active_prompt)

        with st.chat_message("assistant", avatar="🏭"):
            placeholder = st.empty()
            full_response = ""
            start = time.time()
            try:
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=1024,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *[{"role": m["role"], "content": m["content"]}
                          for m in st.session_state["messages"]],
                    ],
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_response += delta
                        placeholder.markdown(full_response + "▌")
                latency = (time.time() - start) * 1000
                placeholder.markdown(full_response)
                st.caption(f"{latency:.0f}ms · llama-3.3-70b-versatile (Groq free tier)")
            except GroqAuthError:
                st.error("Invalid API key. Get a free one at console.groq.com")
                st.stop()
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.session_state["messages"].append({"role": "assistant", "content": full_response})

    if st.session_state.get("messages"):
        if st.button("Clear conversation", type="secondary"):
            st.session_state["messages"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FAILURE AUTOPSIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Failure Autopsies":
    st.markdown("""
    <div style="margin-bottom: 8px;">
      <div style="font-size: 11px; font-weight: 700; letter-spacing: 2px;
                  text-transform: uppercase; color: #4a9eff;">PRODUCTION POST-MORTEMS</div>
      <h1 style="margin: 4px 0 2px 0;">Six Failure Autopsies</h1>
      <div style="opacity: 0.5; font-size: 13px;">
        Real failure classes diagnosed and fixed in a multi-agent procurement platform.
        Each entry shows the symptom, root cause, fix, and code comparison.
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Failures Documented", "6", "All production-verified")
    c2.metric("Max MTTR", "72h → CI gate", "Failure 4 — schema drift")
    c3.metric("Best Latency Fix", "47s → 4.2s", "Failure 2 — async parallelism")

    st.divider()

    sev_badge = {
        "CRITICAL": '<span class="badge badge-critical">CRITICAL</span>',
        "HIGH":     '<span class="badge badge-high">HIGH</span>',
    }

    for f in FAILURES:
        badge = sev_badge.get(f["severity"], "")
        with st.expander(
            f"Failure {f['id']}  ·  {f['title']}  —  {f['subtitle']}",
            expanded=False,
        ):
            # Severity badge
            st.markdown(f"{badge}", unsafe_allow_html=True)
            st.markdown("")

            # Top row: symptom / root cause / fix
            col_s, col_r, col_f = st.columns(3)
            with col_s:
                st.markdown("**Symptom**")
                st.info(f["symptom"])
            with col_r:
                st.markdown("**Root Cause**")
                st.warning(f["root_cause"])
            with col_f:
                st.markdown("**Fix**")
                st.success(f["fix"])

            # Before / After metric
            st.markdown(
                metric_boxes(f["metric_before"], f["metric_after"]),
                unsafe_allow_html=True,
            )

            # Full-width code comparison in tabs
            tab_broken, tab_fixed = st.tabs(["Broken Code", "Fixed Code"])
            with tab_broken:
                st.code(f["broken_code"], language="python")
            with tab_fixed:
                st.code(f["fixed_code"], language="python")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Architecture":
    st.markdown("""
    <div style="margin-bottom: 8px;">
      <div style="font-size: 11px; font-weight: 700; letter-spacing: 2px;
                  text-transform: uppercase; color: #4a9eff;">SYSTEM DESIGN</div>
      <h1 style="margin: 4px 0 2px 0;">Architecture Overview</h1>
      <div style="opacity: 0.5; font-size: 13px;">
        Multi-agent procurement intelligence platform · Meridian Manufacturing Corp
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Visual pipeline diagram
    st.markdown("### Agent Pipeline")
    st.markdown("""
<div style="font-family: monospace; line-height: 1.8; background: #1e2130;
            border: 1px solid #2d3250; border-radius: 12px; padding: 24px; margin-bottom: 24px;">
<div style="color:#4a9eff; font-weight:700; margin-bottom:8px;">USER QUERY</div>
<div style="color:#666;">│</div>
<div style="color:#4a9eff; font-weight:600;">FastAPI  /query  endpoint</div>
<div style="color:#666;">│</div>
<div style="color:#4a9eff; font-weight:600;">LangGraph Stateful Agent</div>
<div style="color:#555; padding-left: 20px;">
  ├── <span style="color:#ccc;">route_query</span> <span style="color:#555;">— intent classification</span><br>
  ├── <span style="color:#ffa500;">summarize_if_needed</span> <span style="color:#555;">— every 8 turns, LLM compression  <span style="color:#ffa500; font-size:11px;">FAILURE 3 FIX</span></span><br>
  └── <span style="color:#ccc;">agent node</span> <span style="color:#555;">— Llama 3.3 70B (Groq)</span>
</div>
<div style="color:#666;">│</div>
<div style="color:#4a9eff; font-weight:600;">Tools Node  (all async)</div>
<div style="color:#555; padding-left: 20px;">
  ├── <span style="color:#ccc;">search_contracts</span>
  <span style="color:#555;"> ──→ Hybrid RAG</span><br>
  <span style="padding-left: 40px; color:#555;">├── Dense:  BGE-M3 embeddings → Qdrant</span><br>
  <span style="padding-left: 40px; color:#555;">├── Sparse: BM25Okapi</span><br>
  <span style="padding-left: 40px; color:#555;">└── RRF Fusion + Table Boost  <span style="color:#ff4b4b; font-size:11px;">FAILURE 1 FIX</span></span><br>
  ├── <span style="color:#ccc;">lookup_supplier</span>
  <span style="color:#555;"> ─────→ SQLite (aiosqlite)</span><br>
  ├── <span style="color:#ccc;">get_live_status</span>
  <span style="color:#555;"> ─────→ Supplier REST API</span>
  <span style="color:#ff4b4b; font-size:11px;"> FAILURE 5 FIX — OAuth2 mutex</span><br>
  ├── <span style="color:#ccc;">flag_risks</span>
  <span style="color:#555;"> ──────────→ asyncio.gather parallel</span>
  <span style="color:#ffa500; font-size:11px;"> FAILURE 2 FIX</span><br>
  └── <span style="color:#ccc;">initiate_approval</span>
  <span style="color:#555;"> ────→ Threshold-based chain routing</span>
</div>
<div style="color:#666; margin-top:12px;">─────────────────────────────────────────────────</div>
<div style="color:#555; margin-top:8px;">
  State:   <span style="color:#ccc;">SQLite checkpointing (LangGraph AsyncSqliteSaver)</span><br>
  Cache:   <span style="color:#ccc;">Redis semantic cache, cosine > 0.92</span>
  <span style="color:#ffa500; font-size:11px;"> FAILURE 2 FIX</span><br>
  Guard:   <span style="color:#ccc;">Schema snapshot → diff → CI exit 1</span>
  <span style="color:#ff4b4b; font-size:11px;"> FAILURE 4 FIX</span><br>
  Evals:   <span style="color:#ccc;">LLM-as-judge + LLM-as-attacker adversarial</span>
  <span style="color:#ffa500; font-size:11px;"> FAILURE 6 FIX</span>
</div>
</div>
""", unsafe_allow_html=True)

    # Tech stack cards
    st.markdown("### Tech Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="stack-card">
  <h4>Agent Layer</h4>
  <div class="stack-row"><span class="stack-key">Orchestration</span><span class="stack-val">LangGraph</span></div>
  <div class="stack-row"><span class="stack-key">LLM</span><span class="stack-val">Claude claude-sonnet-4-6 (Anthropic)</span></div>
  <div class="stack-row"><span class="stack-key">State persistence</span><span class="stack-val">SQLite checkpointing</span></div>
  <div class="stack-row"><span class="stack-key">Tools</span><span class="stack-val">7 async @tool functions</span></div>
</div>
<div class="stack-card">
  <h4>RAG Layer</h4>
  <div class="stack-row"><span class="stack-key">Dense retrieval</span><span class="stack-val">BGE-M3 → Qdrant</span></div>
  <div class="stack-row"><span class="stack-key">Sparse retrieval</span><span class="stack-val">BM25Okapi</span></div>
  <div class="stack-row"><span class="stack-key">Fusion</span><span class="stack-val">Reciprocal Rank Fusion (k=60)</span></div>
  <div class="stack-row"><span class="stack-key">Chunking</span><span class="stack-val">TableAwareChunker (PyMuPDF)</span></div>
</div>
<div class="stack-card">
  <h4>Auth</h4>
  <div class="stack-row"><span class="stack-key">Protocol</span><span class="stack-val">OAuth2 client_credentials</span></div>
  <div class="stack-row"><span class="stack-key">Concurrency</span><span class="stack-val">asyncio.Lock double-checked</span></div>
  <div class="stack-row"><span class="stack-key">Retry</span><span class="stack-val">Exponential backoff, 3 retries</span></div>
  <div class="stack-row"><span class="stack-key">Errors</span><span class="stack-val">Explicit — never silent None</span></div>
</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="stack-card">
  <h4>Data Layer</h4>
  <div class="stack-row"><span class="stack-key">OLTP</span><span class="stack-val">SQLite (aiosqlite)</span></div>
  <div class="stack-row"><span class="stack-key">OLAP</span><span class="stack-val">DuckDB</span></div>
  <div class="stack-row"><span class="stack-key">Cache</span><span class="stack-val">Redis semantic cache</span></div>
  <div class="stack-row"><span class="stack-key">Schema guard</span><span class="stack-val">Snapshot diff + CI gate</span></div>
</div>
<div class="stack-card">
  <h4>Observability</h4>
  <div class="stack-row"><span class="stack-key">Tracing</span><span class="stack-val">LangFuse (self-hosted)</span></div>
  <div class="stack-row"><span class="stack-key">Metrics</span><span class="stack-val">Prometheus + Grafana</span></div>
  <div class="stack-row"><span class="stack-key">Alerts</span><span class="stack-val">4 rules (latency, drift, auth, eval)</span></div>
  <div class="stack-row"><span class="stack-key">API</span><span class="stack-val">FastAPI + /health + /ready</span></div>
</div>
<div class="stack-card">
  <h4>Eval & CI/CD</h4>
  <div class="stack-row"><span class="stack-key">LLM judge</span><span class="stack-val">Faithfulness + relevance scoring</span></div>
  <div class="stack-row"><span class="stack-key">Adversarial</span><span class="stack-val">LLM-as-attacker, 7 attack types</span></div>
  <div class="stack-row"><span class="stack-key">Framework</span><span class="stack-val">DeepEval</span></div>
  <div class="stack-row"><span class="stack-key">CI</span><span class="stack-val">GitHub Actions, 5-stage pipeline</span></div>
</div>
""", unsafe_allow_html=True)

    # CI Pipeline
    st.markdown("### CI/CD Pipeline")
    stages = [
        ("Stage 1", "Schema Drift Check", "python -m src.data.schema_monitor diff", "#ff4b4b"),
        ("Stage 2", "Unit Tests (39 tests)", "pytest tests/ -v", "#4a9eff"),
        ("Stage 3", "Adversarial Eval", "eval_runner --mode adversarial  (gate: ≥90%)", "#ffa500"),
        ("Stage 4", "Faithfulness Eval", "eval_runner --mode faithfulness  (gate: ≥85%)", "#ffa500"),
        ("Stage 5", "Build & Push", "docker build + push to registry", "#21c354"),
    ]
    pipeline_html = '<div style="display:flex; flex-direction:column; gap:6px; margin-top:12px;">'
    for i, (label, name, cmd, color) in enumerate(stages):
        connector = f'<div style="width:2px; height:12px; background:{color}44; margin-left:19px;"></div>' if i < len(stages)-1 else ""
        pipeline_html += f"""
<div style="display:flex; align-items:center; gap:12px;">
  <div style="min-width:38px; height:38px; border-radius:50%; background:{color}22;
              border:2px solid {color}66; display:flex; align-items:center; justify-content:center;
              color:{color}; font-weight:700; font-size:12px;">{i+1}</div>
  <div style="background:#1e2130; border:1px solid #2d3250; border-radius:8px;
              padding:8px 16px; flex:1;">
    <div style="font-size:11px; color:{color}; font-weight:700; letter-spacing:0.5px;">{label} — {name}</div>
    <div style="font-size:11px; font-family:monospace; opacity:0.5; margin-top:2px;">{cmd}</div>
  </div>
</div>{connector}"""
    pipeline_html += "</div>"
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # Supplier table
    st.divider()
    st.markdown("### Supplier Risk Register")
    rows = [
        {
            "ID": sid,
            "Supplier": s["name"],
            "Country": s["country"],
            "Category": s["category"],
            "Risk Score": s["risk_score"],
            "Risk Level": risk_label(s["risk_score"]),
            "On-Time %": f"{s['on_time_delivery_rate']*100:.0f}%",
            "Annual Spend": f"${s['annual_spend_usd']:,.0f}",
            "Status": s["status"].upper(),
        }
        for sid, s in SUPPLIERS.items()
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)
