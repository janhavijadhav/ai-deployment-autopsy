"""
FAILURE MUSEUM — AI System Autopsy Platform
============================================
Live interactive dashboard: inject known failure modes into a running
procurement intelligence pipeline and watch detection + remediation in real time.

Run locally:
    pip install streamlit groq
    streamlit run failure_museum.py

Free Groq key (for Agent Chat): console.groq.com
"""

from __future__ import annotations
import json, re, time, hashlib, sqlite3, tempfile, os, random
from dataclasses import dataclass, field
from typing import Optional
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Failure Museum",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""<style>
#MainMenu,footer{visibility:hidden}
.fm-header{padding:4px 0 20px 0}
.fm-eyebrow{font-size:11px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#4a9eff;margin-bottom:4px}
.fm-title{font-size:2rem;font-weight:800;letter-spacing:-1px;margin:0 0 4px 0;line-height:1.1}
.fm-sub{font-size:13px;opacity:.45}

/* Failure cards */
.fc{border-radius:12px;padding:18px;border:1px solid #2d3250;background:#1a1d2e;margin-bottom:8px;transition:border-color .2s}
.fc-idle{border-color:#2d3250}
.fc-injected{border-color:#ff4b4b;background:#1a1010}
.fc-fixed{border-color:#21c354;background:#0f1a14}
.fc-num{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;font-weight:800;font-size:12px;margin-right:8px}
.fc-num-idle{background:#4a9eff22;color:#4a9eff;border:1px solid #4a9eff44}
.fc-num-injected{background:#ff4b4b22;color:#ff4b4b;border:1px solid #ff4b4b66}
.fc-num-fixed{background:#21c35422;color:#21c354;border:1px solid #21c35444}
.fc-title{font-weight:700;font-size:14px}
.fc-sub{font-size:11px;opacity:.5;margin-top:1px}
.fc-status-idle{font-size:10px;font-weight:700;letter-spacing:1px;color:#4a9eff;opacity:.6}
.fc-status-injected{font-size:10px;font-weight:700;letter-spacing:1px;color:#ff4b4b}
.fc-status-fixed{font-size:10px;font-weight:700;letter-spacing:1px;color:#21c354}

/* Severity badges */
.badge{display:inline-block;padding:2px 9px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:.8px}
.badge-critical{background:#ff4b4b18;color:#ff4b4b;border:1px solid #ff4b4b44}
.badge-high{background:#ffa50018;color:#ffa500;border:1px solid #ffa50044}

/* Chunk display */
.chunk-table{background:#ff4b4b18;border:1px solid #ff4b4b44;border-radius:6px;padding:10px;margin:4px 0;font-size:11px;font-family:monospace;white-space:pre-wrap;word-break:break-all}
.chunk-table-ok{background:#21c35418;border:1px solid #21c35444;border-radius:6px;padding:10px;margin:4px 0;font-size:11px;font-family:monospace;white-space:pre-wrap;word-break:break-all}
.chunk-text{background:#4a9eff10;border:1px solid #4a9eff22;border-radius:6px;padding:10px;margin:4px 0;font-size:11px;font-family:monospace;white-space:pre-wrap;word-break:break-all}
.chunk-label{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px}
.chunk-label-broken{color:#ff4b4b}
.chunk-label-ok{color:#21c354}
.chunk-label-text{color:#4a9eff}

/* Step walkthrough */
.step{display:flex;gap:14px;margin-bottom:14px;align-items:flex-start}
.step-num{min-width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;flex-shrink:0}
.step-pending{background:#2d3250;color:#666;border:2px solid #2d3250}
.step-active{background:#4a9eff22;color:#4a9eff;border:2px solid #4a9eff}
.step-done{background:#21c35422;color:#21c354;border:2px solid #21c354}
.step-error{background:#ff4b4b22;color:#ff4b4b;border:2px solid #ff4b4b}
.step-body{flex:1;padding-top:5px}
.step-title{font-weight:600;font-size:13px}
.step-detail{font-size:11px;opacity:.55;margin-top:2px;font-family:monospace}

/* System health bar */
.health-bar{height:6px;border-radius:3px;margin-top:4px}

/* Log line */
.log-line{font-family:monospace;font-size:11px;padding:2px 0;border-bottom:1px solid #1a1d2e}
.log-ok{color:#21c354}
.log-warn{color:#ffa500}
.log-error{color:#ff4b4b}
.log-info{color:#4a9eff}

/* Metric cards */
.mcard{background:#1a1d2e;border:1px solid #2d3250;border-radius:10px;padding:14px 18px;text-align:center}
.mcard-val{font-size:1.6rem;font-weight:800;letter-spacing:-1px}
.mcard-label{font-size:10px;opacity:.5;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-top:2px}
.mcard-delta-good{color:#21c354;font-size:11px;margin-top:2px}
.mcard-delta-bad{color:#ff4b4b;font-size:11px;margin-top:2px}

/* Worker circles for OAuth2 demo */
.workers{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}
.worker{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:11px;border:2px solid transparent;transition:all .3s}
.worker-idle{background:#2d3250;color:#666;border-color:#3d4260}
.worker-refreshing{background:#ff4b4b22;color:#ff4b4b;border-color:#ff4b4b}
.worker-waiting{background:#4a9eff18;color:#4a9eff;border-color:#4a9eff44}
.worker-done{background:#21c35422;color:#21c354;border-color:#21c354}
.worker-locked{background:#ffa50022;color:#ffa500;border-color:#ffa500}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INLINE IMPLEMENTATIONS (self-contained, no heavy deps)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_CONTRACT = """MASTER SUPPLY AGREEMENT — APEX INDUSTRIES
Contract ID: CTR-00001  |  Effective: 2025-01-01  |  Expires: 2025-12-31

SECTION 4 — DELIVERY AND PERFORMANCE

4.1 On-time delivery is required as defined in Schedule A. Apex Industries
shall notify Meridian Manufacturing of any anticipated delays within 48 hours.

SECTION 9 — LATE DELIVERY PENALTIES

The following penalty schedule applies to all purchase orders under this agreement:

| Delay Period        | Daily Penalty Rate | Maximum Cap         |
|---------------------|--------------------|---------------------|
| Day 1 through 30    | 0.50% per day      | 10% of PO Value     |
| Day 31 through 60   | 0.75% per day      | 15% of PO Value     |
| Day 61 and beyond   | 1.00% per day      | 20% of PO Value     |

Penalties are calculated on the net PO value excluding taxes and freight.
Late payment by Meridian accrues interest at 1.5% per month.

SECTION 12 — QUALITY STANDARDS

All materials must conform to Meridian QA-2024 specification. Rejection rate
above 3% in any rolling 90-day period triggers a corrective action plan.
Repeated non-conformance may result in contract suspension under Section 18."""

def naive_chunk(text: str, chunk_size: int = 300, overlap: int = 40) -> list[dict]:
    """Broken version — splits every N characters. Tables get cut mid-row."""
    chunks, i, idx = [], 0, 0
    while i < len(text):
        content = text[i:i + chunk_size]
        chunks.append({"id": idx, "content": content, "type": "text", "start": i})
        i += chunk_size - overlap
        idx += 1
    return chunks

def table_aware_chunk(text: str, max_words: int = 80) -> list[dict]:
    """Fixed version — tables are atomic, text splits at sentence boundaries."""
    lines = text.split('\n')
    blocks: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_table = False

    for line in lines:
        is_table_line = bool(re.match(r'\s*\|', line)) or (line.count('|') >= 2)
        if is_table_line != in_table:
            if buf:
                blocks.append(('\n'.join(buf), in_table))
            buf = [line]
            in_table = is_table_line
        else:
            buf.append(line)
    if buf:
        blocks.append(('\n'.join(buf), in_table))

    chunks, idx = [], 0
    for block_text, is_table in blocks:
        block_text = block_text.strip()
        if not block_text:
            continue
        if is_table:
            chunks.append({"id": idx, "content": block_text, "type": "TABLE"})
            idx += 1
        else:
            sentences = re.split(r'(?<=[.!?])\s+', block_text)
            current = ""
            for sent in sentences:
                if len((current + " " + sent).split()) > max_words and current:
                    chunks.append({"id": idx, "content": current.strip(), "type": "TEXT"})
                    idx += 1
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent
            if current.strip():
                chunks.append({"id": idx, "content": current.strip(), "type": "TEXT"})
                idx += 1
    return chunks

def table_present(chunk_content: str) -> bool:
    return bool(re.search(r'\|.*\|', chunk_content))

def chunk_is_split_table(content: str) -> bool:
    """Detect a chunk that contains partial table content (broken mid-row)."""
    lines = [l for l in content.split('\n') if l.strip()]
    has_pipe = any('|' in l for l in lines)
    has_full_table = sum(1 for l in lines if l.count('|') >= 3) >= 3
    return has_pipe and not has_full_table

# Schema monitor (inline)
def schema_diff(baseline: dict, current: dict) -> list[dict]:
    changes = []
    for tbl in baseline:
        if tbl not in current:
            changes.append({"change": "TABLE_DROPPED", "table": tbl, "severity": "CRITICAL"})
            continue
        for col, meta in baseline[tbl].items():
            if col not in current[tbl]:
                changes.append({
                    "change": "COLUMN_DROPPED", "table": tbl, "column": col,
                    "severity": "CRITICAL",
                    "impact": f"Queries referencing {tbl}.{col} will return 0 rows silently."
                })
            elif current[tbl][col]["type"] != meta["type"]:
                changes.append({
                    "change": "TYPE_CHANGED", "table": tbl, "column": col,
                    "severity": "HIGH", "was_type": meta["type"], "now_type": current[tbl][col]["type"]
                })
        for col in current.get(tbl, {}):
            if col not in baseline[tbl]:
                changes.append({"change": "COLUMN_ADDED", "table": tbl, "column": col, "severity": "INFO"})
    return changes

def extract_schema(db_path: str) -> dict:
    schema = {}
    conn = sqlite3.connect(db_path)
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    for tbl in tables:
        cols = conn.execute(f"PRAGMA table_info({tbl})").fetchall()
        schema[tbl] = {c[1]: {"type": c[2], "notnull": bool(c[3]), "primary_key": bool(c[5])} for c in cols}
    conn.close()
    return schema

def schema_checksum(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:12]

# ══════════════════════════════════════════════════════════════════════════════
# FAILURE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
FAILURES_META = [
    {"id": 1, "title": "Hallucination Cascade",   "subtitle": "Table-Blind RAG Chunking",      "severity": "CRITICAL", "page": "Chunk Lab"},
    {"id": 2, "title": "Latency Wall",             "subtitle": "Sequential Tool Execution",     "severity": "HIGH",     "page": "Latency Sim"},
    {"id": 3, "title": "Context Collapse",         "subtitle": "Unbounded Conversation State",  "severity": "HIGH",     "page": None},
    {"id": 4, "title": "Schema Drift Bomb",        "subtitle": "Silent Database Migration",     "severity": "CRITICAL", "page": "Schema Drift"},
    {"id": 5, "title": "Auth Deadlock",            "subtitle": "OAuth2 Token Race Condition",   "severity": "HIGH",     "page": "Race Condition"},
    {"id": 6, "title": "Eval Lies",                "subtitle": "Benchmark Overfitting",         "severity": "HIGH",     "page": "Eval Observatory"},
]

# ── Session state defaults ────────────────────────────────────────────────────
def _init():
    for k, v in {
        "failure_states": {i: "idle" for i in range(1, 7)},
        "event_log":      [],
        "messages":       [],
        "pending_query":  None,
        "schema_step":    0,
        "schema_db":      None,
        "schema_baseline": None,
    }.items():
        st.session_state.setdefault(k, v)

_init()

def log(msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    st.session_state["event_log"].insert(0, {"ts": ts, "msg": msg, "level": level})
    st.session_state["event_log"] = st.session_state["event_log"][:50]

def inject(fid: int):
    st.session_state["failure_states"][fid] = "injected"
    f = next(f for f in FAILURES_META if f["id"] == fid)
    log(f"[INJECT] Failure {fid}: {f['title']} activated", "error")

def fix(fid: int):
    st.session_state["failure_states"][fid] = "fixed"
    f = next(f for f in FAILURES_META if f["id"] == fid)
    log(f"[FIX]    Failure {fid}: {f['title']} remediated", "ok")

def reset(fid: int):
    st.session_state["failure_states"][fid] = "idle"

def injected_count() -> int:
    return sum(1 for s in st.session_state["failure_states"].values() if s == "injected")

def health_score() -> int:
    return max(0, 100 - injected_count() * 16)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 12px 0">
      <div class="fm-eyebrow">MERIDIAN MANUFACTURING</div>
      <div style="font-size:17px;font-weight:800;letter-spacing:-.5px">Failure Museum</div>
      <div style="font-size:11px;opacity:.4;margin-top:2px">AI System Autopsy Platform</div>
    </div>""", unsafe_allow_html=True)

    # System health
    h = health_score()
    hcol = "#21c354" if h >= 80 else "#ffa500" if h >= 50 else "#ff4b4b"
    st.markdown(f"""
    <div style="background:#1a1d2e;border:1px solid #2d3250;border-radius:8px;padding:10px 14px;margin-bottom:12px">
      <div style="font-size:10px;opacity:.5;letter-spacing:1px;font-weight:700;text-transform:uppercase">System Health</div>
      <div style="font-size:22px;font-weight:800;color:{hcol};letter-spacing:-1px">{h}%</div>
      <div class="health-bar" style="background:linear-gradient(90deg,{hcol} {h}%,#2d3250 {h}%)"></div>
    </div>""", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "Injection Console",
        "Chunk Lab",
        "Reranker Lab",
        "Agent Delegation",
        "Latency Simulator",
        "Schema Drift",
        "Race Condition",
        "Eval Observatory",
        "Agent Chat",
    ], label_visibility="collapsed")

    st.divider()
    # Failure status mini-list
    st.markdown("<div style='font-size:10px;opacity:.4;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px'>FAILURE STATUS</div>", unsafe_allow_html=True)
    icons = {"idle": "○", "injected": "●", "fixed": "✓"}
    colors = {"idle": "#4a9eff55", "injected": "#ff4b4b", "fixed": "#21c354"}
    for f in FAILURES_META:
        s = st.session_state["failure_states"][f["id"]]
        st.markdown(
            f'<div style="font-size:11px;display:flex;justify-content:space-between;padding:2px 0">'
            f'<span style="opacity:.7">F{f["id"]} {f["title"]}</span>'
            f'<span style="color:{colors[s]};font-weight:700">{icons[s]}</span></div>',
            unsafe_allow_html=True)

    st.divider()
    api_key = st.secrets.get("GROQ_API_KEY", "")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INJECTION CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Injection Console":
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">FAILURE MUSEUM — CONTROL ROOM</div>
      <div class="fm-title">Injection Console</div>
      <div class="fm-sub">Select any failure mode to inject into the live pipeline. Watch detection fire and remediation apply.</div>
    </div>""", unsafe_allow_html=True)

    # Global metrics
    n_injected = injected_count()
    n_fixed = sum(1 for s in st.session_state["failure_states"].values() if s == "fixed")
    mc = st.columns(4)
    mc[0].metric("System Health", f"{health_score()}%", f"-{n_injected*16}%" if n_injected else "Nominal")
    mc[1].metric("Active Failures", n_injected, "DEGRADED" if n_injected else "Clean")
    mc[2].metric("Remediated", n_fixed)
    mc[3].metric("Est. P95 Latency", f"{4.2 + n_injected * 10.7:.0f}s", f"+{n_injected * 10.7:.0f}s" if n_injected else "Nominal")

    st.divider()

    badge_html = {"CRITICAL": '<span class="badge badge-critical">CRITICAL</span>',
                  "HIGH":     '<span class="badge badge-high">HIGH</span>'}

    col_a, col_b = st.columns(2)
    for i, f in enumerate(FAILURES_META):
        col = col_a if i % 2 == 0 else col_b
        state = st.session_state["failure_states"][f["id"]]
        css = {"idle": "fc-idle", "injected": "fc-injected", "fixed": "fc-fixed"}[state]
        num_css = {"idle": "fc-num-idle", "injected": "fc-num-injected", "fixed": "fc-num-fixed"}[state]
        status_css = {"idle": "fc-status-idle", "injected": "fc-status-injected", "fixed": "fc-status-fixed"}[state]
        status_txt = {"idle": "IDLE", "injected": "● INJECTED", "fixed": "✓ FIXED"}[state]

        with col:
            st.markdown(f"""
            <div class="fc {css}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
                <div style="display:flex;align-items:center">
                  <div class="fc-num {num_css}">{f["id"]}</div>
                  <div>
                    <div class="fc-title">{f["title"]}</div>
                    <div class="fc-sub">{f["subtitle"]}</div>
                  </div>
                </div>
                <div>{badge_html[f["severity"]]}</div>
              </div>
              <div class="{status_css}">{status_txt}</div>
            </div>""", unsafe_allow_html=True)

            b1, b2, b3 = st.columns(3)
            if b1.button("Inject", key=f"inj_{f['id']}", type="primary",
                         disabled=state == "injected", use_container_width=True):
                inject(f["id"]); st.rerun()
            if b2.button("Fix", key=f"fix_{f['id']}",
                         disabled=state != "injected", use_container_width=True):
                fix(f["id"]); st.rerun()
            if b3.button("Reset", key=f"rst_{f['id']}",
                         disabled=state == "idle", use_container_width=True):
                reset(f["id"]); st.rerun()

            if f["page"]:
                st.caption(f"→ Live demo in **{f['page']}** page")
            st.markdown("")

    # Event log
    st.divider()
    st.markdown("**System Event Log**")
    if not st.session_state["event_log"]:
        st.caption("No events yet. Inject a failure to begin.")
    else:
        log_html = ""
        for e in st.session_state["event_log"][:15]:
            css = {"ok": "log-ok", "error": "log-error", "warn": "log-warn", "info": "log-info"}.get(e["level"], "log-info")
            log_html += f'<div class="log-line {css}">[{e["ts"]}] {e["msg"]}</div>'
        st.markdown(log_html, unsafe_allow_html=True)
    if st.button("Clear log"):
        st.session_state["event_log"] = []; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHUNK LAB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Chunk Lab":
    state = st.session_state["failure_states"][1]
    st.markdown(f"""
    <div class="fm-header">
      <div class="fm-eyebrow">FAILURE 1 — LIVE DEMO</div>
      <div class="fm-title">Chunk Analysis Lab</div>
      <div class="fm-sub">Watch the naive chunker destroy a penalty table. Watch the fix preserve it.</div>
    </div>""", unsafe_allow_html=True)

    status_color = {"idle": "#4a9eff", "injected": "#ff4b4b", "fixed": "#21c354"}[state]
    status_text  = {"idle": "IDLE — inject to activate broken mode",
                    "injected": "FAILURE ACTIVE — naive chunker engaged",
                    "fixed": "FIXED — table-aware chunker active"}[state]
    st.markdown(f'<div style="background:{status_color}18;border:1px solid {status_color}44;border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:12px;color:{status_color};font-weight:700">{status_text}</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    if b1.button("Inject Failure", type="primary", disabled=state=="injected"):
        inject(1); st.rerun()
    if b2.button("Apply Fix", disabled=state!="injected"):
        fix(1); st.rerun()
    if b3.button("Reset"):
        reset(1); st.rerun()

    st.divider()

    contract_text = st.text_area("Contract Text (edit to test your own)", SAMPLE_CONTRACT, height=200)
    chunk_size = st.slider("Naive chunk size (chars)", 100, 600, 300, 50)

    if st.button("Run Both Chunkers", type="primary"):
        with st.spinner("Running chunkers..."):
            time.sleep(0.4)
            naive  = naive_chunk(contract_text, chunk_size=chunk_size)
            smart  = table_aware_chunk(contract_text)

        st.markdown("")
        left, right = st.columns(2)

        with left:
            st.markdown("### Naive Chunker (BROKEN)")
            st.markdown(f"**{len(naive)} chunks** · chunk_size={chunk_size}")
            table_splits = sum(1 for c in naive if chunk_is_split_table(c["content"]))
            if table_splits:
                st.markdown(f'<div style="color:#ff4b4b;font-size:12px;font-weight:700">⚠ Table split across {table_splits} chunk(s) — LLM will hallucinate values</div>', unsafe_allow_html=True)
            for c in naive:
                has_pipe = table_present(c["content"])
                is_split = chunk_is_split_table(c["content"])
                if is_split:
                    css, label_css, label = "chunk-table", "chunk-label-broken", "SPLIT TABLE — BROKEN"
                elif has_pipe:
                    css, label_css, label = "chunk-table", "chunk-label-broken", "PARTIAL TABLE"
                else:
                    css, label_css, label = "chunk-text", "chunk-label-text", f"TEXT chunk #{c['id']}"
                st.markdown(
                    f'<div class="{css}"><div class="chunk-label {label_css}">{label}</div>{c["content"][:300]}{"…" if len(c["content"])>300 else ""}</div>',
                    unsafe_allow_html=True)

        with right:
            st.markdown("### Table-Aware Chunker (FIXED)")
            st.markdown(f"**{len(smart)} chunks** · atomic tables")
            table_chunks = [c for c in smart if c["type"] == "TABLE"]
            st.markdown(f'<div style="color:#21c354;font-size:12px;font-weight:700">✓ {len(table_chunks)} table chunk(s) preserved atomically</div>', unsafe_allow_html=True)
            for c in smart:
                if c["type"] == "TABLE":
                    css, label_css, label = "chunk-table-ok", "chunk-label-ok", "TABLE — ATOMIC (intact)"
                else:
                    css, label_css, label = "chunk-text", "chunk-label-text", f"TEXT chunk #{c['id']}"
                st.markdown(
                    f'<div class="{css}"><div class="chunk-label {label_css}">{label}</div>{c["content"][:300]}{"…" if len(c["content"])>300 else ""}</div>',
                    unsafe_allow_html=True)

        # Score comparison
        st.divider()
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Naive chunks", len(naive))
        sc2.metric("Smart chunks", len(smart))
        sc3.metric("Table integrity", "0%" if table_splits else "100%",
                   "BROKEN" if table_splits else "OK",
                   delta_color="inverse" if table_splits else "normal")

        naive_halluc = 23 if table_splits else 5
        smart_halluc = 2
        st.markdown(f"""
        <div style="display:flex;gap:12px;margin-top:16px">
          <div style="flex:1;background:#ff4b4b18;border:1px solid #ff4b4b44;border-radius:8px;padding:14px">
            <div style="font-size:10px;color:#ff4b4b;font-weight:700;letter-spacing:1px">NAIVE — HALLUCINATION RATE</div>
            <div style="font-size:2rem;font-weight:800;color:#ff4b4b;letter-spacing:-1px">{naive_halluc}%</div>
          </div>
          <div style="flex:1;background:#21c35418;border:1px solid #21c35444;border-radius:8px;padding:14px">
            <div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1px">TABLE-AWARE — HALLUCINATION RATE</div>
            <div style="font-size:2rem;font-weight:800;color:#21c354;letter-spacing:-1px">{smart_halluc}%</div>
          </div>
        </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RERANKER LAB
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Reranker Lab":
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">TECH STACK — RAG STAGE 3</div>
      <div class="fm-title">Reranker Lab</div>
      <div class="fm-sub">Cross-encoder reranking fixes the keyword-collision failure mode in hybrid retrieval. Run the live demo to see it in action.</div>
    </div>""", unsafe_allow_html=True)

    # ── Explainer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:16px;margin-bottom:24px">
      <div style="flex:1;background:#4a9eff10;border:1px solid #4a9eff33;border-radius:10px;padding:16px">
        <div style="font-size:10px;color:#4a9eff;font-weight:700;letter-spacing:1px;margin-bottom:8px">RETRIEVAL PIPELINE — 3 STAGES</div>
        <div style="font-size:12px;line-height:1.9;font-family:monospace">
          <span style="color:#4a9eff">1.</span> Dense (BGE-M3) + Sparse (BM25) → RRF Fusion<br>
          <span style="color:#4a9eff">2.</span> Table Boost (×1.25 for structured data)<br>
          <span style="color:#ff9900;font-weight:700">3.</span> <span style="color:#ff9900;font-weight:700">Cross-Encoder Rerank (BAAI/bge-reranker-v2-m3)</span>
        </div>
      </div>
      <div style="flex:1;background:#ff4b4b10;border:1px solid #ff4b4b33;border-radius:10px;padding:16px">
        <div style="font-size:10px;color:#ff4b4b;font-weight:700;letter-spacing:1px;margin-bottom:8px">WHY STAGE 1+2 ALONE FAILS</div>
        <div style="font-size:12px;line-height:1.6">Bi-encoders (BGE-M3) embed query and document <em>independently</em> — they never attend to each other. BM25 is pure keyword overlap. A query for <strong>"beyond 60 days"</strong> matches the <strong>"Day 31–60"</strong> chunk's keyword "60" — wrong answer, high rank.</div>
      </div>
      <div style="flex:1;background:#21c35410;border:1px solid #21c35433;border-radius:10px;padding:16px">
        <div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1px;margin-bottom:8px">WHY CROSS-ENCODER FIXES IT</div>
        <div style="font-size:12px;line-height:1.6">A cross-encoder concatenates <em>(query, document)</em> as a single input — every query token attends to every document token. It trivially understands that <strong>"beyond 60 days"</strong> means <strong>Day 61+</strong>, not Day 31–60.</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # ── The concrete failure scenario ─────────────────────────────────────────
    st.markdown("#### The Classic Keyword-Collision Scenario")
    st.markdown("""
    <div style="background:#1a1d2e;border:1px solid #2d3250;border-radius:8px;padding:14px;margin-bottom:16px;font-size:13px">
      <span style="color:#4a9eff;font-weight:700">Query:</span>
      <span style="font-family:monospace;background:#4a9eff18;border-radius:4px;padding:2px 8px;margin-left:6px">"What is the penalty for delivery delays <strong>beyond 60 days</strong>?"</span>
    </div>""", unsafe_allow_html=True)

    RERANK_CHUNKS = [
        {
            "chunk_id": "C1",
            "label": "Day 31–60 Penalties",
            "content": "Late delivery penalty schedule — Day 31 through 60: 1.5% of total invoice value per calendar week of delay, compounded weekly.",
            "rrf_rank": 1,
            "rrf_score": 0.0323,
            "why_rrf_high": 'BM25 keyword "60" collision — this chunk ranks #1 despite being the WRONG answer',
            "ce_score": -0.84,
        },
        {
            "chunk_id": "C2",
            "label": "Day 61+ Penalties",
            "content": "Escalated delay penalties — Day 61 and beyond: 3.0% of total invoice value per calendar week, plus Meridian retains the right to terminate the contract with immediate effect and recover all consequential losses.",
            "rrf_rank": 2,
            "rrf_score": 0.0298,
            "why_rrf_high": "Correct answer — contains 'beyond' — but BM25 scores lower than 'Day 31-60' chunk",
            "ce_score": 2.71,
        },
        {
            "chunk_id": "C3",
            "label": "Grace Period Clause",
            "content": "A 5-business-day grace period applies to all deliveries before any late penalty accrues. Grace periods do not stack across multiple delayed shipments.",
            "rrf_rank": 3,
            "rrf_score": 0.0251,
            "why_rrf_high": "Contextually related to delivery delays",
            "ce_score": 0.33,
        },
        {
            "chunk_id": "C4",
            "label": "Force Majeure Waiver",
            "content": "Penalties are waived during force majeure events as defined in Section 18. Supplier must notify Meridian within 48 hours of the triggering event.",
            "rrf_rank": 4,
            "rrf_score": 0.0214,
            "why_rrf_high": "Contains 'penalty' — keyword overlap",
            "ce_score": -1.22,
        },
        {
            "chunk_id": "C5",
            "label": "Payment Terms",
            "content": "Standard payment terms: Net-60. Early payment discount of 2% applies if settled within 10 days of invoice receipt.",
            "rrf_rank": 5,
            "rrf_score": 0.0187,
            "why_rrf_high": 'BM25 matched "60" in "Net-60" — pure noise',
            "ce_score": -2.89,
        },
    ]

    # Sort by cross-encoder score for the "after" view
    ce_sorted = sorted(RERANK_CHUNKS, key=lambda x: x["ce_score"], reverse=True)

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("""<div style="font-size:10px;color:#ff4b4b;font-weight:700;letter-spacing:1px;margin-bottom:10px">BEFORE RERANKING — RRF ORDER</div>""", unsafe_allow_html=True)
        for i, chunk in enumerate(RERANK_CHUNKS):
            is_correct = chunk["chunk_id"] == "C2"
            border_color = "#21c35444" if is_correct else "#2d3250"
            bg_color = "#21c35408" if is_correct else "#1a1d2e"
            rank_color = "#ff4b4b" if i == 0 and not is_correct else ("#21c354" if is_correct else "#4a9eff88")
            st.markdown(f"""
            <div style="background:{bg_color};border:1px solid {border_color};border-radius:8px;padding:12px;margin-bottom:8px">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
                <div>
                  <span style="font-size:10px;font-weight:800;color:{rank_color};letter-spacing:1px">RANK #{chunk['rrf_rank']}</span>
                  <span style="font-size:11px;font-weight:700;margin-left:8px">{chunk['label']}</span>
                </div>
                <span style="font-size:10px;font-family:monospace;color:#4a9eff88">RRF {chunk['rrf_score']:.4f}</span>
              </div>
              <div style="font-size:11px;opacity:.7;line-height:1.5;margin-bottom:6px">{chunk['content'][:120]}…</div>
              <div style="font-size:10px;color:{"#ff4b4b" if "WRONG" in chunk["why_rrf_high"] or "noise" in chunk["why_rrf_high"] else "#4a9eff88"};font-style:italic">{chunk['why_rrf_high']}</div>
            </div>""", unsafe_allow_html=True)

    with col_after:
        st.markdown("""<div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1px;margin-bottom:10px">AFTER RERANKING — CROSS-ENCODER ORDER</div>""", unsafe_allow_html=True)
        for i, chunk in enumerate(ce_sorted):
            is_correct = chunk["chunk_id"] == "C2"
            is_promoted = is_correct and i == 0
            border_color = "#21c35488" if is_correct else "#2d3250"
            bg_color = "#21c35412" if is_correct else "#1a1d2e"
            score_color = "#21c354" if chunk["ce_score"] > 0 else "#ff4b4b"
            st.markdown(f"""
            <div style="background:{bg_color};border:1px solid {border_color};border-radius:8px;padding:12px;margin-bottom:8px{';box-shadow:0 0 12px #21c35422' if is_promoted else ''}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px">
                <div>
                  <span style="font-size:10px;font-weight:800;color:{'#21c354' if is_correct else '#4a9eff88'};letter-spacing:1px">RANK #{i+1}{'  PROMOTED' if is_promoted else ''}</span>
                  <span style="font-size:11px;font-weight:700;margin-left:8px">{chunk['label']}</span>
                </div>
                <span style="font-size:10px;font-family:monospace;color:{score_color};font-weight:700">CE {chunk['ce_score']:+.2f}</span>
              </div>
              <div style="font-size:11px;opacity:.7;line-height:1.5">{chunk['content'][:120]}…</div>
              {'<div style="font-size:10px;color:#21c354;font-weight:700;margin-top:6px">Correct answer now at rank #1</div>' if is_promoted else ''}
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Live interactive demo ─────────────────────────────────────────────────
    st.markdown("#### Try Your Own Query")
    st.markdown('<div style="font-size:12px;opacity:.5;margin-bottom:10px">Enter a query to see Jaccard-based simulation (no model weights needed) rerank these 5 contract chunks in real time.</div>', unsafe_allow_html=True)

    user_query = st.text_input(
        "Query",
        value="What is the penalty for delivery delays beyond 60 days?",
        label_visibility="collapsed",
    )

    if user_query:
        import re as _re

        def _tokenize(text):
            return set(_re.findall(r'\b\w+\b', text.lower()))

        def _jaccard_rerank(query, chunks):
            qtoks = _tokenize(query)
            scored = []
            for c in chunks:
                ctoks = _tokenize(c["content"])
                j = len(qtoks & ctoks) / len(qtoks | ctoks) if (qtoks | ctoks) else 0
                scored.append({**c, "sim_ce_score": round((j * 6.0) - 3.0, 3)})
            return sorted(scored, key=lambda x: x["sim_ce_score"], reverse=True)

        reranked_live = _jaccard_rerank(user_query, RERANK_CHUNKS)
        rrf_order_ids = [c["chunk_id"] for c in RERANK_CHUNKS]
        ce_order_ids  = [c["chunk_id"] for c in reranked_live]

        # Show side-by-side comparison
        l, r = st.columns(2)
        with l:
            st.markdown('<div style="font-size:10px;color:#ff4b4b;font-weight:700;letter-spacing:1px;margin-bottom:8px">RRF ORDER (FIXED)</div>', unsafe_allow_html=True)
            for i, cid in enumerate(rrf_order_ids):
                chunk = next(c for c in RERANK_CHUNKS if c["chunk_id"] == cid)
                st.markdown(f'<div style="font-size:12px;padding:7px 10px;border-radius:6px;background:#1a1d2e;border:1px solid #2d3250;margin:3px 0">#{i+1} — {chunk["label"]}</div>', unsafe_allow_html=True)

        with r:
            st.markdown('<div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1px;margin-bottom:8px">RERANKED ORDER (LIVE)</div>', unsafe_allow_html=True)
            for i, chunk in enumerate(reranked_live):
                rrf_pos = rrf_order_ids.index(chunk["chunk_id"]) + 1
                moved = rrf_pos - (i + 1)
                arrow = f' <span style="color:#21c354">▲{moved}</span>' if moved > 0 else (f' <span style="color:#ff4b4b">▼{abs(moved)}</span>' if moved < 0 else '')
                score_color = "#21c354" if chunk["sim_ce_score"] > 0 else "#ff4b4b"
                st.markdown(f'<div style="font-size:12px;padding:7px 10px;border-radius:6px;background:#1a1d2e;border:1px solid #2d3250;margin:3px 0;display:flex;justify-content:space-between">#{i+1} — {chunk["label"]}{arrow}<span style="font-family:monospace;color:{score_color};font-size:11px">{chunk["sim_ce_score"]:+.3f}</span></div>', unsafe_allow_html=True)

    st.divider()

    # ── Architecture note ─────────────────────────────────────────────────────
    st.markdown("#### Model Card: BAAI/bge-reranker-v2-m3")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Size", "568 MB")
    m2.metric("Languages", "100+")
    m3.metric("Max Tokens", "512")
    m4.metric("Latency (CPU)", "~90ms / 20 pairs")

    st.markdown("""
    <div style="background:#1a1d2e;border:1px solid #2d3250;border-radius:8px;padding:14px;margin-top:8px;font-size:12px;line-height:1.7">
      <strong>Pipeline position:</strong> Runs on the top <code>top_k × 4 = 20</code> RRF candidates — bounded latency while maximising precision.<br>
      <strong>Async:</strong> <code>rerank_async()</code> runs in a thread pool via <code>loop.run_in_executor()</code> so it never blocks the event loop.<br>
      <strong>Graceful degradation:</strong> If the model is unavailable (no weights, OOM), retriever automatically falls back to RRF-only — zero downtime.<br>
      <strong>Observability:</strong> Both <code>cross_encoder_score</code> and original <code>rrf_score</code> are stored in chunk metadata for LangFuse comparison.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENT DELEGATION
# Inline routing logic (mirrors src/agent/supervisor.py) — self-contained so
# the Streamlit demo runs without installing heavy LangGraph/Anthropic deps.
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Agent Delegation":
    # ── Inline specialist registry (mirrors src/agent/supervisor.SPECIALISTS) ──
    _SPECS = {
        "contract_analyst": {
            "name": "Contract Analyst", "color": "#4a9eff", "icon": "📄",
            "desc": "Contract clauses, SLA terms, penalty schedules, renewal dates",
            "tools": ["search_contracts", "get_contract_metadata"],
            "kw": ["contract","clause","sla","penalty","penalt","terms","renewal",
                   "expir","terminat","amend","payment","warranty","late","delay",
                   "net-","force majeure","liability","indemnif"],
        },
        "supplier_risk": {
            "name": "Supplier Risk", "color": "#ff4b4b", "icon": "⚠️",
            "desc": "Risk scoring, delivery performance, geopolitical exposure, disputes",
            "tools": ["lookup_supplier","search_suppliers_by_criteria","flag_supplier_risks"],
            "kw": ["risk","supplier","vendor","delivery","performance","on-time","on time",
                   "probation","flag","alert","exposure","sanction","certif","audit",
                   "compliance","quality","rejection","dispute","apex","brightfield",
                   "dalton","coretech","pinnacle","sup-","tier-"],
        },
        "spend_analytics": {
            "name": "Spend Analytics", "color": "#21c354", "icon": "📊",
            "desc": "Spend analysis, savings opportunities, budget forecasting, KPIs",
            "tools": ["get_procurement_analytics"],
            "kw": ["spend","saving","budget","forecast","trend","analytics","cost",
                   "price","quarter","annual","q1","q2","q3","q4","ytd","monthly",
                   "kpi","report","total","rebate","discount","opportunit","dashboard"],
        },
    }

    def _classify(query):
        q = query.lower()
        hits = {}
        for sid, sp in _SPECS.items():
            m = [kw for kw in sp["kw"] if kw in q]
            if m:
                hits[sid] = m
        if not hits:
            return {"specialists":["contract_analyst"],"primary":"contract_analyst",
                    "reasoning":"No strong domain signal — defaulting to Contract Analyst.",
                    "confidence":0.45,"multi_agent":False,"keywords_matched":[]}
        ranked = sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)
        top = len(ranked[0][1])
        sel = [s for s,kws in ranked if len(kws) >= max(1, top//2)]
        all_kw = [kw for _,kws in ranked for kw in kws]
        primary = ranked[0][0]
        if len(sel)==1:
            spec = _SPECS[primary]
            _kw_str = ", ".join(f'"{k}"' for k in hits[primary][:4])
            return {"specialists":sel,"primary":primary,
                    "reasoning": (f"Query clearly targets the {spec['name']} domain. "
                                  f"Matched keywords: {_kw_str}. "
                                  f"Single-specialist delegation — no synthesis needed."),
                    "confidence": min(0.65+len(hits[primary])*0.06, 0.97),
                    "multi_agent":False,"keywords_matched":all_kw}
        names = " + ".join(_SPECS[s]["name"] for s in sel)
        _kw_str2 = ", ".join(f'"{k}"' for k in all_kw[:5])
        return {"specialists":sel,"primary":primary,
                "reasoning": (f"Cross-domain query — signals from {len(sel)} specialist domains. "
                              f"Delegating to {names} in parallel. "
                              f"Matched: {_kw_str2}. "
                              f"Synthesizer will merge outputs."),
                "confidence": min(0.70+len(all_kw)*0.03, 0.96),
                "multi_agent":True,"keywords_matched":all_kw}

    # ── Inline simulation responses ─────────────────────────────────────────────
    def _sim_contract(q):
        if any(w in q for w in ["penalt","delay","late"]):
            return ("**Contract Analyst — Penalty Clause Analysis**\n\n"
                    "Source: CTR-00001 (Apex Industries), Section 8.2\n\n"
                    "| Window | Rate |\n|---|---|\n"
                    "| Days 1–30 | 0.5% / week |\n"
                    "| Days 31–60 | 1.5% / week (compounded) |\n"
                    "| Day 61+ | 3.0% / week + right to terminate |\n\n"
                    "**Risk Note:** Day 61+ includes consequential loss recovery (§8.2.4) — "
                    "non-standard for Tier-2. Flagging for next renewal review.",
                    ["CTR-00001 §8.2","CTR-00001 §8.2.4"],["search_contracts"])
        elif any(w in q for w in ["expir","renew","terminat"]):
            return ("**Contract Analyst — Expiry & Renewal Report**\n\n"
                    "• **CTR-00001** (Apex) — Expires 2024-12-31. Auto-renewal §14.1. "
                    "Notice window opens **Sep 1** — action required.\n"
                    "• **CTR-00003** (Dalton) — Expires 2024-06-30 ⚠️ Probation — renewal not recommended.\n"
                    "• **CTR-00002** (Brightfield) — Expires 2025-03-31. Normal cycle.",
                    ["CTR-00001 §14.1","CTR-00003 §14.1"],["search_contracts","get_contract_metadata"])
        elif any(w in q for w in ["payment","net","invoice"]):
            return ("**Contract Analyst — Payment Terms**\n\n"
                    "| Supplier | Terms | Early Pay |\n|---|---|---|\n"
                    "| Apex (CTR-00001) | Net-60 | 2% if Net-10 |\n"
                    "| Brightfield (CTR-00002) | Net-45 | None |\n"
                    "| Dalton (CTR-00003) | Net-30 | None |\n\n"
                    "Savings insight: capturing Apex early-pay = **$568K/year** (§9.1.3).",
                    ["CTR-00001 §9.1","CTR-00002 §9.1"],["search_contracts"])
        return ("**Contract Analyst — Clause Search**\n\n"
                "• CTR-00001 §4.3: Force majeure covers disruption > 14 days, 48h notification.\n"
                "• CTR-00002 §7.1: SLA uptime 99.5%/month — breach triggers 10% invoice credit.\n"
                "• CTR-00001 §11.2: Custom tooling IP reverts to Meridian on contract end.",
                ["CTR-00001 §4.3","CTR-00002 §7.1"],["search_contracts"])

    def _sim_risk(q):
        if any(w in q for w in ["apex","sup-0001"]):
            return ("**Supplier Risk — Apex Industries (SUP-0001)**\n\n"
                    "Risk Score: **0.82** 🔴 CRITICAL\n\n"
                    "1. Delivery 87% (SLA 95%) — Q3 worst in 3 quarters\n"
                    "2. CN geopolitical exposure — Section 301 tariff review active\n"
                    "3. Financial: revenue −12% YoY, D/E ratio 2.3× (threshold 1.5×)\n\n"
                    "**Action: ESCALATE** — Activate dual-sourcing. Brief CPO in 48h.",
                    ["SUP-0001","RISK-SUP-0001-HIGH"],["lookup_supplier","flag_supplier_risks"])
        elif any(w in q for w in ["dalton","probation","sup-0003"]):
            return ("**Supplier Risk — Dalton Materials (SUP-0003)**\n\n"
                    "Risk Score: **0.67** 🟡 HIGH | Status: PROBATION\n\n"
                    "1. Quality rejection 9.0% (threshold 2.0%) — 4.5× above limit\n"
                    "2. 4 open disputes ($1.2M total)\n"
                    "3. On-time delivery 71% — worst in portfolio\n\n"
                    "**Action: SUSPEND NEW POs** pending Q3 quality audit (Sep 30).",
                    ["SUP-0003","RISK-SUP-0003-QUALITY"],["lookup_supplier","flag_supplier_risks"])
        return ("**Supplier Risk — Portfolio Overview**\n\n"
                "| Supplier | Risk | Tier | On-Time |\n|---|---|---|---|\n"
                "| Apex Industries | 0.82 | 🔴 CRITICAL | 87% |\n"
                "| Dalton Materials | 0.67 | 🟡 HIGH | 71% |\n"
                "| Pinnacle Logistics | 0.44 | 🟡 MOD | 89% |\n"
                "| CoreTech Systems | 0.33 | 🟢 LOW | 93% |\n"
                "| Brightfield | 0.21 | 🟢 LOW | 97% |\n\n"
                "At-risk spend: **$37.5M** (Apex $28.4M + Dalton $9.1M)",
                ["SUP-0001","SUP-0003"],["search_suppliers_by_criteria","flag_supplier_risks"])

    def _sim_spend(q):
        if any(w in q for w in ["saving","opportunit","rebate","discount"]):
            return ("**Spend Analytics — Savings Opportunities**\n\n"
                    "Total addressable: **$4.2M** (1.75% of spend)\n\n"
                    "1. Early pay capture (Apex) — **$568K**\n"
                    "2. Electronics volume consolidation — **$1.8M**\n"
                    "3. Pinnacle rebate trigger ($1.3M to threshold) — **$890K**\n"
                    "4. Dalton replacement net savings — **$960K**",
                    ["SAVINGS-Q3-2024"],["get_procurement_analytics"])
        elif any(w in q for w in ["spend","budget","ytd","total","annual"]):
            return ("**Spend Analytics — YTD 2024**\n\n"
                    "Total: **$1.82B** / $2.4B budget (76%)\n\n"
                    "| Supplier | YTD | YoY |\n|---|---|---|\n"
                    "| Brightfield | $41.2M | ↑8% |\n"
                    "| Pinnacle | $28.7M | ↑3% |\n"
                    "| Apex | $23.1M | ↓14% |\n"
                    "| CoreTech | $15.2M | →0% |\n"
                    "| Dalton | $6.9M | ↓24% |\n\n"
                    "Full-year forecast: **$2.35B** (2% under budget)",
                    ["SPEND-YTD-2024"],["get_procurement_analytics"])
        return ("**Spend Analytics — 30-Day KPIs**\n\n"
                "| KPI | Value |\n|---|---|\n"
                "| Total spend | $152M |\n| POs issued | 1,247 |\n"
                "| Contract coverage | 94.2% ↑ |\n| Savings rate | 1.38% ↑ |\n"
                "| On-time payment | 96.4% |\n\n"
                "Insight: Contract coverage improving — spot buy down 2.3pp.",
                ["KPI-LAST-30-DAYS"],["get_procurement_analytics"])

    _SIM_FNS = {
        "contract_analyst": _sim_contract,
        "supplier_risk": _sim_risk,
        "spend_analytics": _sim_spend,
    }

    # ── Page header ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">TECH STACK — LANGGRAPH DEPTH</div>
      <div class="fm-title">Agent Delegation</div>
      <div class="fm-sub">Supervisor agent classifies every query and routes it to the right specialist — or fans out to multiple in parallel for cross-domain questions.</div>
    </div>""", unsafe_allow_html=True)

    # ── Architecture diagram (small independent calls — avoids Streamlit HTML escaping) ──
    st.markdown('<div style="font-size:10px;color:#4a9eff;font-weight:700;letter-spacing:1.5px;margin-bottom:8px">LANGGRAPH TOPOLOGY</div>', unsafe_allow_html=True)
    _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns([2, 1, 3, 1, 2])
    with _dc1:
        st.markdown('<div style="background:#4a9eff18;border:1.5px solid #4a9eff55;border-radius:8px;padding:12px;text-align:center"><div style="font-size:9px;color:#4a9eff;font-weight:700;letter-spacing:1px">START</div><div style="font-size:13px;font-weight:700;margin-top:4px">User Query</div></div>', unsafe_allow_html=True)
    with _dc2:
        st.markdown('<div style="text-align:center;padding-top:18px;font-size:20px;color:#4a9eff;opacity:.7">→</div>', unsafe_allow_html=True)
    with _dc3:
        st.markdown('<div style="background:#ff990018;border:2px solid #ff9900;border-radius:8px;padding:12px;text-align:center;margin-bottom:8px"><div style="font-size:9px;color:#ff9900;font-weight:700;letter-spacing:1px">SUPERVISOR NODE</div><div style="font-size:13px;font-weight:700;margin-top:4px">classify_query()</div><div style="font-size:10px;opacity:.5;margin-top:2px">picks specialist(s) · returns Send()</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:10px;opacity:.4;text-align:center;margin-bottom:6px">↓ conditional_edges → Send()</div>', unsafe_allow_html=True)
        _sc1, _sc2, _sc3 = st.columns(3)
        _sc1.markdown('<div style="background:#4a9eff18;border:1px solid #4a9eff55;border-radius:6px;padding:8px;text-align:center"><div style="font-size:9px;color:#4a9eff;font-weight:700">📄 CONTRACT</div><div style="font-size:9px;opacity:.5;margin-top:2px">clauses · SLA</div></div>', unsafe_allow_html=True)
        _sc2.markdown('<div style="background:#ff4b4b18;border:1px solid #ff4b4b55;border-radius:6px;padding:8px;text-align:center"><div style="font-size:9px;color:#ff4b4b;font-weight:700">⚠️ RISK</div><div style="font-size:9px;opacity:.5;margin-top:2px">risk · delivery</div></div>', unsafe_allow_html=True)
        _sc3.markdown('<div style="background:#21c35418;border:1px solid #21c35455;border-radius:6px;padding:8px;text-align:center"><div style="font-size:9px;color:#21c354;font-weight:700">📊 SPEND</div><div style="font-size:9px;opacity:.5;margin-top:2px">spend · KPIs</div></div>', unsafe_allow_html=True)
    with _dc4:
        st.markdown('<div style="text-align:center;padding-top:18px;font-size:20px;color:#4a9eff;opacity:.7">→</div>', unsafe_allow_html=True)
    with _dc5:
        st.markdown('<div style="background:#21c35418;border:1.5px solid #21c35455;border-radius:8px;padding:12px;text-align:center"><div style="font-size:9px;color:#21c354;font-weight:700;letter-spacing:1px">SYNTHESIZER NODE</div><div style="font-size:13px;font-weight:700;margin-top:4px">Merge + END</div><div style="font-size:10px;opacity:.5;margin-top:2px">multi-specialist only</div></div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Example queries ─────────────────────────────────────────────────────────
    EXAMPLE_QUERIES = [
        ("What are Apex's penalty clauses?",                       "contract_analyst"),
        ("Show all high-risk suppliers above 0.7",                 "supplier_risk"),
        ("What's our YTD spend and savings opportunities?",        "spend_analytics"),
        ("Apex penalty terms AND their risk score — full picture", "contract_analyst+supplier_risk"),
        ("Contract expiry, risk scores, and Q4 spend forecast",    "all three"),
    ]

    st.markdown("#### Run the Supervisor")
    ex_cols = st.columns(len(EXAMPLE_QUERIES))
    for i, (ex_q, _) in enumerate(EXAMPLE_QUERIES):
        if ex_cols[i].button(ex_q[:30]+"…" if len(ex_q)>30 else ex_q,
                             key=f"del_ex_{i}", use_container_width=True, help=ex_q):
            st.session_state["del_query"] = ex_q

    user_del_q = st.text_input(
        "Or type your own query",
        value=st.session_state.get("del_query", "What are Apex's penalty clauses for late delivery?"),
        key="del_query_input",
        label_visibility="collapsed",
        placeholder="Ask anything about contracts, suppliers, spend…",
    )

    if user_del_q:
        decision = _classify(user_del_q)
        specialists = decision["specialists"]
        primary = decision["primary"]
        multi = decision["multi_agent"]

        # ── Step 1: Supervisor decision ────────────────────────────────────────
        st.markdown("---")
        conf_color = "#21c354" if decision["confidence"]>0.75 else "#ff9900" if decision["confidence"]>0.55 else "#ff4b4b"
        st.markdown(f"""
        <div style="background:#ff990012;border:1px solid #ff990044;border-radius:10px;padding:16px;margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div>
              <span style="font-size:10px;color:#ff9900;font-weight:700;letter-spacing:1.5px">STEP 1 — SUPERVISOR NODE</span>
              <span style="margin-left:10px;font-size:11px;background:#ff990022;color:#ff9900;padding:2px 8px;border-radius:10px;font-weight:700">{"MULTI-AGENT FAN-OUT" if multi else "SINGLE SPECIALIST"}</span>
            </div>
            <span style="font-size:11px;font-family:monospace;color:{conf_color};font-weight:700">confidence {decision['confidence']:.0%}</span>
          </div>
          <div style="font-size:12px;line-height:1.6;opacity:.85">{decision['reasoning']}</div>
          <div style="margin-top:10px;display:flex;gap:8px;flex-wrap:wrap">
            {"".join(f'<span style="background:{_SPECS[s]["color"]}22;color:{_SPECS[s]["color"]};border:1px solid {_SPECS[s]["color"]}55;border-radius:12px;padding:3px 10px;font-size:11px;font-weight:700">{_SPECS[s]["icon"]} {_SPECS[s]["name"]}</span>' for s in specialists)}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Step 2: Specialist responses ──────────────────────────────────────
        st.markdown(f"""<div style="font-size:10px;color:#4a9eff;font-weight:700;letter-spacing:1.5px;margin-bottom:8px">
        STEP 2 — SPECIALIST NODE{"S (PARALLEL)" if multi else ""}</div>""", unsafe_allow_html=True)

        results = {}
        q_lower = user_del_q.lower()
        if multi:
            cols = st.columns(len(specialists))
            for i, sid in enumerate(specialists):
                ans, srcs, tools = _SIM_FNS[sid](q_lower)
                results[sid] = ans
                spec = _SPECS[sid]
                with cols[i]:
                    st.markdown(f"""
                    <div style="background:{spec['color']}0d;border:1px solid {spec['color']}44;border-radius:8px;padding:12px;margin-bottom:8px">
                      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                        <span style="font-size:10px;color:{spec['color']};font-weight:700;letter-spacing:1px">{spec['icon']} {spec['name'].upper()}</span>
                        <span style="font-size:9px;font-family:monospace;color:{spec['color']}88">{len(tools)} tool(s)</span>
                      </div>
                      <div style="font-size:10px;opacity:.55;margin-bottom:6px">Tools: {', '.join(f'<code>{t}</code>' for t in tools)}</div>
                      <div style="font-size:10px;opacity:.55">Sources: {' · '.join(srcs[:2])}</div>
                    </div>""", unsafe_allow_html=True)
                    with st.expander(f"View {spec['name']} response"):
                        st.markdown(ans)
        else:
            sid = specialists[0]
            ans, srcs, tools = _SIM_FNS[sid](q_lower)
            results[sid] = ans
            spec = _SPECS[sid]
            st.markdown(f"""
            <div style="background:{spec['color']}0d;border:1px solid {spec['color']}44;border-radius:8px;padding:14px;margin-bottom:10px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:10px;color:{spec['color']};font-weight:700;letter-spacing:1px">{spec['icon']} {spec['name'].upper()} — ACTIVE</span>
                <span style="font-size:10px;font-family:monospace;color:{spec['color']}88">Tools: {', '.join(tools)}</span>
              </div>
              <div style="font-size:11px;opacity:.6">Sources cited: {' · '.join(srcs)}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(ans)

        # ── Step 3: Synthesizer ────────────────────────────────────────────────
        if multi:
            st.markdown("---")
            st.markdown("""<div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1.5px;margin-bottom:8px">STEP 3 — SYNTHESIZER NODE</div>""", unsafe_allow_html=True)
            names = " + ".join(_SPECS[s]["name"] for s in specialists)
            header = f"*Cross-domain analysis — {len(specialists)} specialists consulted in parallel: {names}*\n\n"
            merged_parts = []
            for sid in specialists:
                merged_parts.append(f"**{_SPECS[sid]['name']}:**\n\n{results[sid]}")
            merged = header + "\n\n---\n\n".join(merged_parts)
            with st.container():
                st.markdown(f"""
                <div style="background:#21c35410;border:1px solid #21c35444;border-radius:8px;padding:12px;margin-bottom:8px">
                  <span style="font-size:10px;color:#21c354;font-weight:700">Synthesizing {len(specialists)} specialist outputs into unified response</span>
                </div>""", unsafe_allow_html=True)
                st.markdown(merged)
        else:
            st.markdown("---")
            st.markdown("""<div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1.5px;margin-bottom:6px">STEP 3 — SYNTHESIZER NODE (pass-through)</div>""", unsafe_allow_html=True)
            st.markdown('<div style="font-size:11px;opacity:.5;margin-bottom:8px">Single specialist — synthesizer passes response through unchanged. No merge step needed.</div>', unsafe_allow_html=True)

        # ── Delegation trace ───────────────────────────────────────────────────
        st.divider()
        with st.expander("View delegation trace (what LangFuse captures)"):
            trace = [
                {"node": "supervisor", "decision": specialists, "confidence": f"{decision['confidence']:.2f}",
                 "multi_agent": multi, "keywords": decision["keywords_matched"][:5]},
            ]
            for sid in specialists:
                trace.append({"node": f"{sid}_node", "specialist": _SPECS[sid]["name"],
                              "tools_called": _SIM_FNS[sid](q_lower)[2],
                              "sources": _SIM_FNS[sid](q_lower)[1][:2]})
            trace.append({"node": "synthesizer", "specialists_merged": specialists, "output": "final_response"})
            st.json(trace)

    st.divider()
    # ── Code insight ────────────────────────────────────────────────────────────
    st.markdown("#### How Multi-Specialist Fan-out Works")
    st.code("""# src/agent/multi_agent_graph.py

def route_after_supervisor(state: MultiAgentState) -> list[Send]:
    \"\"\"
    Conditional edge after supervisor_node.
    Returns Send objects — LangGraph runs them in PARALLEL
    and auto-joins state before synthesizer fires.
    \"\"\"
    specialists = state["supervisor_decision"]["specialists"]
    node_map = {
        "contract_analyst": "contract_analyst_node",
        "supplier_risk":    "supplier_risk_node",
        "spend_analytics":  "spend_analytics_node",
    }
    # Each Send() gets a COPY of state — no locking needed
    return [Send(node_map[s], state) for s in specialists]

# Graph wiring:
graph.add_conditional_edges("supervisor", route_after_supervisor,
    ["contract_analyst_node", "supplier_risk_node", "spend_analytics_node"])

# All specialists converge at synthesizer (LangGraph handles the join)
graph.add_edge("contract_analyst_node", "synthesizer")
graph.add_edge("supplier_risk_node",    "synthesizer")
graph.add_edge("spend_analytics_node",  "synthesizer")
""", language="python")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LATENCY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Latency Simulator":
    state = st.session_state["failure_states"][2]
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">FAILURE 2 — LIVE DEMO</div>
      <div class="fm-title">Latency Simulator</div>
      <div class="fm-sub">Sequential vs parallel tool execution. Watch the latency wall appear — and collapse.</div>
    </div>""", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    if b1.button("Inject Failure", type="primary", disabled=state=="injected"):
        inject(2); st.rerun()
    if b2.button("Apply Fix", disabled=state!="injected"):
        fix(2); st.rerun()
    if b3.button("Reset", disabled=state=="idle"):
        reset(2); st.rerun()

    st.divider()

    n_suppliers = st.slider("Number of suppliers to query", 1, 15, 5)
    lookup_ms   = st.slider("Simulated lookup time per supplier (ms)", 500, 8000, 3000, 500)
    rag_ms      = st.slider("RAG search time (ms)", 500, 10000, 5000, 500)
    risk_ms     = st.slider("Risk flagging time (ms)", 500, 5000, 2000, 500)

    seq_total = rag_ms + (lookup_ms * n_suppliers) + risk_ms
    par_total = rag_ms + lookup_ms + risk_ms
    cache_ms  = 85

    if st.button("Run Simulation", type="primary"):
        left, right = st.columns(2)

        with left:
            st.markdown("#### Broken — Sequential")
            bars = st.empty()
            timer_disp = st.empty()
            elapsed = 0
            steps = (
                [("RAG search", rag_ms)] +
                [(f"Lookup SUP-{i:04d}", lookup_ms) for i in range(1, n_suppliers+1)] +
                [("Flag risks", risk_ms)]
            )
            bar_data = []
            for label, dur in steps:
                elapsed += dur
                bar_data.append({"Task": label, "Time (ms)": dur})
                bars.bar_chart(bar_data, x="Task", y="Time (ms)", color="#ff4b4b", use_container_width=True, height=220)
                timer_disp.markdown(f'<div style="font-size:1.8rem;font-weight:800;color:#ff4b4b;letter-spacing:-1px">{elapsed/1000:.1f}s elapsed</div>', unsafe_allow_html=True)
                time.sleep(min(dur / 4000, 0.35))
            st.markdown(f'<div style="background:#ff4b4b18;border:1px solid #ff4b4b44;border-radius:8px;padding:12px;margin-top:8px"><div style="color:#ff4b4b;font-weight:700">P95 TOTAL: {seq_total/1000:.1f}s</div><div style="font-size:11px;opacity:.6;margin-top:2px">Each lookup blocks the next. Unusable at scale.</div></div>', unsafe_allow_html=True)

        with right:
            st.markdown("#### Fixed — Parallel (asyncio.gather)")
            bars2  = st.empty()
            timer2 = st.empty()
            par_steps = (
                [("RAG search", rag_ms)] +
                [(f"SUP-{i:04d} (parallel)", lookup_ms) for i in range(1, n_suppliers+1)] +
                [("Flag risks", risk_ms)]
            )
            # Animate RAG first, then all lookups at once
            bar_data2 = [{"Task": "RAG search", "Time (ms)": rag_ms}]
            bars2.bar_chart(bar_data2, x="Task", y="Time (ms)", color="#4a9eff", use_container_width=True, height=220)
            timer2.markdown(f'<div style="font-size:1.8rem;font-weight:800;color:#4a9eff;letter-spacing:-1px">{rag_ms/1000:.1f}s elapsed</div>', unsafe_allow_html=True)
            time.sleep(min(rag_ms / 4000, 0.4))
            # All lookups fire simultaneously
            for i in range(1, n_suppliers+1):
                bar_data2.append({"Task": f"SUP-{i:04d}", "Time (ms)": lookup_ms})
            bars2.bar_chart(bar_data2, x="Task", y="Time (ms)", color="#4a9eff", use_container_width=True, height=220)
            elapsed2 = rag_ms + lookup_ms
            timer2.markdown(f'<div style="font-size:1.8rem;font-weight:800;color:#4a9eff;letter-spacing:-1px">{elapsed2/1000:.1f}s elapsed</div>', unsafe_allow_html=True)
            time.sleep(min(lookup_ms / 4000, 0.4))
            bar_data2.append({"Task": "Flag risks", "Time (ms)": risk_ms})
            bars2.bar_chart(bar_data2, x="Task", y="Time (ms)", color="#4a9eff", use_container_width=True, height=220)
            elapsed2 += risk_ms
            timer2.markdown(f'<div style="font-size:1.8rem;font-weight:800;color:#4a9eff;letter-spacing:-1px">{elapsed2/1000:.1f}s elapsed</div>', unsafe_allow_html=True)
            time.sleep(0.2)
            st.markdown(f'<div style="background:#21c35418;border:1px solid #21c35444;border-radius:8px;padding:12px;margin-top:8px"><div style="color:#21c354;font-weight:700">P95 TOTAL: {par_total/1000:.1f}s</div><div style="font-size:11px;opacity:.6;margin-top:2px">All supplier lookups fire simultaneously via asyncio.gather.</div></div>', unsafe_allow_html=True)

        st.divider()
        speedup = seq_total / par_total
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sequential", f"{seq_total/1000:.1f}s")
        c2.metric("Parallel", f"{par_total/1000:.1f}s", f"-{(seq_total-par_total)/1000:.1f}s")
        c3.metric("Speedup", f"{speedup:.1f}×")
        c4.metric("Cache hit", f"{cache_ms}ms", f"-{(par_total-cache_ms)/1000:.1f}s vs parallel")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SCHEMA DRIFT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Schema Drift":
    state = st.session_state["failure_states"][4]
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">FAILURE 4 — LIVE DEMO</div>
      <div class="fm-title">Schema Drift Detector</div>
      <div class="fm-sub">Walk step-by-step through how a silent column rename broke 72 hours of production — and how the CI gate catches it before deploy.</div>
    </div>""", unsafe_allow_html=True)

    step = st.session_state["schema_step"]
    db_path = st.session_state["schema_db"]
    baseline = st.session_state["schema_baseline"]

    STEPS = [
        "Create database",
        "View initial schema",
        "Take snapshot",
        "Deploy migration (rename supplier_id → supplier_code)",
        "Run application query",
        "Run schema diff",
        "See the CI gate fire",
    ]

    def step_css(i):
        if i < step: return "step-done"
        if i == step: return "step-active"
        return "step-pending"

    # Step timeline
    timeline_html = ""
    for i, s in enumerate(STEPS):
        css = step_css(i)
        icon = "✓" if i < step else str(i+1) if i > step else "→"
        timeline_html += f'<div class="step"><div class="step-num {css}">{icon}</div><div class="step-body"><div class="step-title">{s}</div></div></div>'
    st.markdown(timeline_html, unsafe_allow_html=True)

    st.divider()

    # Step actions
    if step == 0:
        if st.button("Create Database", type="primary"):
            tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
            tmp.close()
            conn = sqlite3.connect(tmp.name)
            conn.executescript("""
                CREATE TABLE suppliers (
                    supplier_id   TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    country       TEXT NOT NULL,
                    risk_score    REAL DEFAULT 0.0,
                    status        TEXT DEFAULT 'active'
                );
                CREATE TABLE contracts (
                    contract_id  TEXT PRIMARY KEY,
                    supplier_id  TEXT NOT NULL REFERENCES suppliers(supplier_id),
                    value_usd    REAL NOT NULL
                );
                INSERT INTO suppliers VALUES ('SUP-0001','Apex Industries','CN',0.82,'active');
                INSERT INTO suppliers VALUES ('SUP-0002','Brightfield Components','DE',0.21,'active');
            """)
            conn.commit(); conn.close()
            st.session_state["schema_db"] = tmp.name
            inject(4)
            st.session_state["schema_step"] = 1
            log("Database created with 2 tables, 2 suppliers", "info")
            st.rerun()

    elif step == 1:
        schema = extract_schema(db_path)
        st.markdown("**Current Schema**")
        st.json(schema)
        if st.button("Take Snapshot", type="primary"):
            st.session_state["schema_baseline"] = schema
            st.session_state["schema_step"] = 2
            log(f"Schema snapshot taken — checksum {schema_checksum(schema)}", "ok")
            st.rerun()

    elif step == 2:
        st.success(f"Snapshot saved — checksum `{schema_checksum(baseline)}`")
        st.json(baseline)
        if st.button("Deploy Migration", type="primary"):
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE suppliers_new (
                    supplier_code TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    country       TEXT NOT NULL,
                    risk_score    REAL DEFAULT 0.0,
                    status        TEXT DEFAULT 'active'
                );
                INSERT INTO suppliers_new SELECT supplier_id,name,country,risk_score,status FROM suppliers;
                DROP TABLE suppliers;
                ALTER TABLE suppliers_new RENAME TO suppliers;
            """)
            conn.commit(); conn.close()
            st.session_state["schema_step"] = 3
            log("Migration deployed: supplier_id → supplier_code", "warn")
            st.rerun()

    elif step == 3:
        st.warning("Migration executed. `supplier_id` renamed to `supplier_code`. Application code not updated.")
        if st.button("Run Application Query", type="primary"):
            st.session_state["schema_step"] = 4
            log("Query: SELECT * FROM suppliers WHERE supplier_id = 'SUP-0001'", "info")
            log("Result: 0 rows — no error raised", "error")
            st.rerun()

    elif step == 4:
        st.error("**0 rows returned. No exception. No log entry. No alert.**")
        st.code("""
-- Application code (unchanged):
SELECT * FROM suppliers WHERE supplier_id = ?
-- Params: ('SUP-0001',)

-- Result:
-- (no rows)
-- sqlite3 returns empty cursor silently
-- Agent returns "Supplier not found" to every user""", language="sql")
        if st.button("Run Schema Diff", type="primary"):
            st.session_state["schema_step"] = 5
            log("Schema diff initiated against baseline", "info")
            st.rerun()

    elif step == 5:
        current_schema = extract_schema(db_path)
        changes = schema_diff(baseline, current_schema)
        st.markdown("**Schema diff output:**")
        output_lines = []
        for c in changes:
            if c["severity"] == "CRITICAL":
                output_lines.append(f'<div class="log-line log-error">[CRITICAL] {c["change"]}: {c["table"]}.{c.get("column","")}</div>')
                if "impact" in c:
                    output_lines.append(f'<div class="log-line log-error">           Impact: {c["impact"]}</div>')
            elif c["severity"] == "HIGH":
                output_lines.append(f'<div class="log-line log-warn">[HIGH]     {c["change"]}: {c["table"]}.{c.get("column","")}</div>')
            else:
                output_lines.append(f'<div class="log-line log-info">[INFO]     {c["change"]}: {c["table"]}.{c.get("column","")}</div>')
        output_lines.append('<div class="log-line log-error">Drift detected. Exiting with code 1.</div>')
        st.markdown('\n'.join(output_lines), unsafe_allow_html=True)
        if changes:
            st.session_state["schema_step"] = 6
            log("Schema diff: CRITICAL drift detected — exit code 1", "error")
            st.rerun()

    elif step == 6:
        st.error("CI gate fires — deployment blocked.")
        st.code("""
# .github/workflows/eval-gate.yml
- name: Schema drift check
  run: python -m src.data.schema_monitor diff
  # Exit code: 1 ← CRITICAL drift
  # GitHub Actions marks step FAILED
  # PR cannot merge to main
  # Zero production impact""", language="yaml")
        st.success("In production without this gate: **72 hours downtime, $1.2M in delayed approvals.**")
        fix(4)
        col1, col2 = st.columns(2)
        if col1.button("Restore Database"):
            if db_path and os.path.exists(db_path):
                os.unlink(db_path)
            st.session_state.update({"schema_step": 0, "schema_db": None, "schema_baseline": None})
            log("Database restored to clean state", "ok")
            st.rerun()
        if col2.button("Start Over"):
            st.session_state.update({"schema_step": 0, "schema_db": None, "schema_baseline": None})
            reset(4); st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RACE CONDITION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Race Condition":
    state = st.session_state["failure_states"][5]
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">FAILURE 5 — LIVE DEMO</div>
      <div class="fm-title">OAuth2 Race Condition</div>
      <div class="fm-sub">Ten coroutines. One expired token. Watch the broken version cause a 429 storm — and the mutex fix collapse it to a single refresh call.</div>
    </div>""", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    if b1.button("Inject Failure", type="primary", disabled=state=="injected"):
        inject(5); st.rerun()
    if b2.button("Apply Fix", disabled=state!="injected"):
        fix(5); st.rerun()
    if b3.button("Reset", disabled=state=="idle"):
        reset(5); st.rerun()

    st.divider()

    def render_workers(worker_states: list[str]) -> str:
        labels = {"idle": "W", "waiting": "WAIT", "locked": "LOCK", "refreshing": "REF", "done": "OK", "error": "429"}
        css    = {"idle": "worker-idle", "waiting": "worker-waiting", "locked": "worker-locked",
                  "refreshing": "worker-refreshing", "done": "worker-done", "error": "worker-refreshing"}
        html = '<div class="workers">'
        for i, s in enumerate(worker_states):
            html += f'<div class="worker {css[s]}">{i+1}<br><span style="font-size:8px">{labels[s]}</span></div>'
        html += '</div>'
        return html

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Broken — No Lock")
        st.code("""async def _get_valid_token(self):
    if self._token.is_expired():
        await self._refresh_token()
    # All 10 coroutines hit this simultaneously""", language="python")
        run_broken = st.button("Run Broken Version", type="primary", key="run_broken")

    with col_r:
        st.markdown("#### Fixed — asyncio.Lock()")
        st.code("""async def _get_valid_token(self):
    if self._token and not self._token.is_expired():
        return self._token.access_token
    async with self._refresh_lock:
        if not self._token.is_expired():
            return  # double-check
        await self._refresh_token()""", language="python")
        run_fixed = st.button("Run Fixed Version", type="primary", key="run_fixed")

    st.divider()
    worker_display = st.empty()
    event_display  = st.empty()
    result_display = st.empty()

    if run_broken:
        events = []
        frames = [
            (["idle"]*10, "Token valid..."),
            (["refreshing"]*10, "Token expired — all 10 coroutines detect it simultaneously"),
            (["refreshing"]*10, "All 10 fire _refresh_token() concurrently"),
            (["error"]*9 + ["done"]*1, "Token endpoint 429s 9 of them — only 1 succeeds"),
            (["error"]*9 + ["done"]*1, "9 coroutines enter exponential backoff (1s, 2s, 4s...)"),
        ]
        for workers, msg in frames:
            worker_display.markdown(render_workers(workers), unsafe_allow_html=True)
            events.insert(0, msg)
            event_display.markdown('\n'.join(f'<div class="log-line log-{"error" if "429" in msg or "backoff" in msg else "warn"}">{m}</div>' for m in events[:5]), unsafe_allow_html=True)
            time.sleep(0.7)
        result_display.markdown("""
        <div style="background:#ff4b4b18;border:1px solid #ff4b4b44;border-radius:8px;padding:16px;margin-top:8px">
          <div style="color:#ff4b4b;font-weight:700;font-size:14px">RESULT: Storm</div>
          <div style="font-size:12px;margin-top:6px">9 of 10 requests failed with 429. Total recovery time: <strong>~90 seconds</strong> of cascading backoff. Affected all concurrent users simultaneously.</div>
        </div>""", unsafe_allow_html=True)

    if run_fixed:
        events = []
        frames = [
            (["idle"]*10, "Token valid — all coroutines take fast path (no lock)"),
            (["waiting"]*10, "Token expired — all 10 detect it"),
            (["locked"]*1 + ["waiting"]*9, "Coroutine 1 acquires asyncio.Lock — others queue"),
            (["refreshing"]*1 + ["waiting"]*9, "Coroutine 1 refreshes token (single network call)"),
            (["done"]*1 + ["waiting"]*9, "Fresh token available — lock released"),
            (["done"]*10, "Coroutines 2–10 see valid token on double-check — skip refresh"),
        ]
        for workers, msg in frames:
            worker_display.markdown(render_workers(workers), unsafe_allow_html=True)
            events.insert(0, msg)
            event_display.markdown('\n'.join(f'<div class="log-line log-{"ok" if "done" in workers[-1] or "valid" in msg else "info"}">{m}</div>' for m in events[:6]), unsafe_allow_html=True)
            time.sleep(0.65)
        result_display.markdown("""
        <div style="background:#21c35418;border:1px solid #21c35444;border-radius:8px;padding:16px;margin-top:8px">
          <div style="color:#21c354;font-weight:700;font-size:14px">RESULT: Single refresh</div>
          <div style="font-size:12px;margin-top:6px">1 network call total. All 10 coroutines got a valid token. Total time: <strong>~4.8 seconds</strong>. Zero 429s. Zero user impact.</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVAL OBSERVATORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Eval Observatory":
    state = st.session_state["failure_states"][6]
    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">FAILURE 6 — LIVE DEMO</div>
      <div class="fm-title">Eval Observatory</div>
      <div class="fm-sub">Your eval suite says 94%. Production says 61%. The gap is the failure. Watch adversarial testing close it.</div>
    </div>""", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    if b1.button("Inject Failure", type="primary", disabled=state=="injected"):
        inject(6); st.rerun()
    if b2.button("Apply Fix", disabled=state!="injected"):
        fix(6); st.rerun()
    if b3.button("Reset", disabled=state=="idle"):
        reset(6); st.rerun()

    st.divider()

    CLEAN_CASES = [
        ("What are the penalty clauses for Apex Industries?", "PASS", 0.96),
        ("Show suppliers with risk score above 0.7",          "PASS", 0.94),
        ("When does contract CTR-00001 expire?",              "PASS", 0.97),
        ("What is Brightfield's on-time delivery rate?",      "PASS", 0.93),
        ("List suppliers on probation status",                "PASS", 0.91),
    ]

    ADVERSARIAL_CASES = [
        ("penalti clouses for apex",                                       "typo",             "FAIL", 0.41),
        ("¿Cuáles son las penalizaciones por retraso?",                    "multilingual",     "FAIL", 0.38),
        ("Which suppliers from same country as highest-risk vendor have expiring contracts?", "multi_hop", "FAIL", 0.29),
        ("What's Apex's 99% on-time delivery rate?",                       "wrong_assumption", "FAIL", 0.22),
        ("check the contract",                                             "ambiguous",        "FAIL", 0.31),
        ("yo what's the deal with dalton",                                 "informal",         "FAIL", 0.44),
        ("penalty for late del",                                           "truncated",        "PASS", 0.72),
    ]

    ADVERSARIAL_FIXED = [
        ("penalti clouses for apex",                                       "typo",             "PASS", 0.88),
        ("¿Cuáles son las penalizaciones por retraso?",                    "multilingual",     "PASS", 0.91),
        ("Which suppliers from same country as highest-risk vendor have expiring contracts?", "multi_hop", "PASS", 0.84),
        ("What's Apex's 99% on-time delivery rate?",                       "wrong_assumption", "PASS", 0.87),
        ("check the contract",                                             "ambiguous",        "PASS", 0.83),
        ("yo what's the deal with dalton",                                 "informal",         "PASS", 0.89),
        ("penalty for late del",                                           "truncated",        "PASS", 0.92),
    ]

    adv_cases = ADVERSARIAL_FIXED if state == "fixed" else ADVERSARIAL_CASES
    clean_rate = sum(1 for c in CLEAN_CASES if c[1] == "PASS") / len(CLEAN_CASES)
    adv_rate   = sum(1 for c in adv_cases if c[2] == "PASS") / len(adv_cases)
    prod_score = 0.89 if state == "fixed" else 0.61

    c1, c2, c3 = st.columns(3)
    c1.metric("Clean Eval Pass Rate", f"{clean_rate*100:.0f}%", "Always high — misleading")
    c2.metric("Adversarial Pass Rate", f"{adv_rate*100:.0f}%",
              "+30%" if state == "fixed" else "Below gate (need ≥90%)", delta_color="normal" if state=="fixed" else "inverse")
    c3.metric("Production Satisfaction", f"{prod_score*100:.0f}%",
              "+28% after fix" if state == "fixed" else "Real signal — much lower",
              delta_color="normal" if state=="fixed" else "inverse")

    st.divider()
    left, right = st.columns(2)

    with left:
        st.markdown("**Clean Eval Cases** (always pass — not the signal)")
        for query, verdict, score in CLEAN_CASES:
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 10px;background:#21c35412;border-radius:6px;margin:3px 0;font-size:12px"><span style="opacity:.8">{query}</span><span style="color:#21c354;font-weight:700">{verdict} {score:.0%}</span></div>', unsafe_allow_html=True)

    with right:
        st.markdown(f"**Adversarial Eval Cases** ({'FIXED' if state=='fixed' else 'BROKEN — exposes the gap'})")
        for query, attack, verdict, score in adv_cases:
            color = "#21c354" if verdict == "PASS" else "#ff4b4b"
            st.markdown(f'<div style="display:flex;justify-content:space-between;padding:6px 10px;background:{color}12;border-radius:6px;margin:3px 0;font-size:12px"><div><div style="opacity:.8">{query}</div><div style="font-size:10px;opacity:.45;margin-top:2px">[{attack}]</div></div><span style="color:{color};font-weight:700;white-space:nowrap;margin-left:8px">{verdict} {score:.0%}</span></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown(f"""
    <div style="display:flex;gap:16px">
      <div style="flex:1;background:#ff4b4b18;border:1px solid #ff4b4b44;border-radius:10px;padding:16px">
        <div style="font-size:10px;color:#ff4b4b;font-weight:700;letter-spacing:1px">THE PROBLEM</div>
        <div style="font-size:13px;margin-top:8px;line-height:1.6">Clean eval: <strong>94%</strong> → teams optimised for it every sprint.<br>Production users: <strong>61%</strong> satisfied.<br>Gap widened because the benchmark didn't reflect reality.</div>
      </div>
      <div style="flex:1;background:#21c35418;border:1px solid #21c35444;border-radius:10px;padding:16px">
        <div style="font-size:10px;color:#21c354;font-weight:700;letter-spacing:1px">THE FIX</div>
        <div style="font-size:13px;margin-top:8px;line-height:1.6">LLM-as-attacker generates 50 adversarial cases per CI run across 7 attack types.<br>Gate: adversarial pass rate <strong>≥ 90%</strong>. Cannot merge below it.</div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AGENT CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Agent Chat":
    SUPPLIERS_DATA = {
        "SUP-0001": {"name":"Apex Industries","country":"CN","risk_score":0.82,"annual_spend_usd":28400000,"on_time_delivery_rate":0.87,"quality_rejection_rate":0.04,"open_disputes":1,"status":"active","certifications":["ISO-9001","ISO-14001"]},
        "SUP-0002": {"name":"Brightfield Components","country":"DE","risk_score":0.21,"annual_spend_usd":54200000,"on_time_delivery_rate":0.97,"quality_rejection_rate":0.01,"open_disputes":0,"status":"active","certifications":["ISO-9001","IATF-16949","AS9100"]},
        "SUP-0003": {"name":"Dalton Materials","country":"MX","risk_score":0.67,"annual_spend_usd":9100000,"on_time_delivery_rate":0.71,"quality_rejection_rate":0.09,"open_disputes":4,"status":"probation","certifications":["ISO-9001"]},
        "SUP-0004": {"name":"CoreTech Systems","country":"US","risk_score":0.33,"annual_spend_usd":18700000,"on_time_delivery_rate":0.93,"quality_rejection_rate":0.02,"open_disputes":0,"status":"active"},
        "SUP-0005": {"name":"Pinnacle Logistics","country":"SG","risk_score":0.44,"annual_spend_usd":31500000,"on_time_delivery_rate":0.89,"quality_rejection_rate":0.00,"open_disputes":2,"status":"active"},
    }
    SYSTEM_PROMPT = f"""You are the Procurement Intelligence Agent for Meridian Manufacturing Corp ($2.4B spend, 1,200+ contracts).
Help with supplier risk, contract terms, PO approvals, spend analytics.
Risk rules: >0.70 = HIGH (VP+CPO for POs>$500K), 0.40-0.70 = MEDIUM, <0.40 = LOW (auto-approve <$2M).
Approval: <$100K=ProcMgr, $100K-$1M=VP, >$1M=CFO+VP.
SUPPLIER DATA: {json.dumps(SUPPLIERS_DATA)}"""

    st.markdown("""
    <div class="fm-header">
      <div class="fm-eyebrow">LIVE AGENT</div>
      <div class="fm-title">Procurement Intelligence Agent</div>
      <div class="fm-sub">Powered by Llama 3.3 70B via Groq free tier. Ask about suppliers, contracts, risks, or approvals.</div>
    </div>""", unsafe_allow_html=True)

    examples = [
        "What are the penalty clauses for Apex Industries?",
        "Which suppliers have risk scores above 0.7 and open disputes?",
        "Approve a $750K PO for Dalton Materials — what's the approval chain?",
        "Compare on-time delivery rates across all 5 suppliers.",
    ]
    with st.expander("Example queries"):
        cols = st.columns(2)
        for i, ex in enumerate(examples):
            if cols[i%2].button(ex, key=f"chat_ex_{i}", use_container_width=True):
                st.session_state["messages"].append({"role": "user", "content": ex})
                st.session_state["pending_query"] = ex
                st.rerun()

    # ── Inline supervisor for chat routing badges ──────────────────────────────
    _CHAT_SPECS = {
        "contract_analyst": {"name":"Contract Analyst","color":"#4a9eff","icon":"📄",
            "kw":["contract","clause","sla","penalty","penalt","terms","renewal","expir",
                  "terminat","amend","payment","warranty","late","delay","net-","force majeure"]},
        "supplier_risk":    {"name":"Supplier Risk","color":"#ff4b4b","icon":"⚠️",
            "kw":["risk","supplier","vendor","delivery","performance","on-time","probation",
                  "flag","alert","exposure","quality","rejection","dispute","apex","brightfield",
                  "dalton","coretech","pinnacle","sup-"]},
        "spend_analytics":  {"name":"Spend Analytics","color":"#21c354","icon":"📊",
            "kw":["spend","saving","budget","forecast","trend","analytics","cost","quarter",
                  "annual","ytd","kpi","report","total","rebate","discount","opportunit"]},
    }
    def _chat_route(query):
        q = query.lower()
        hits = {sid: [kw for kw in sp["kw"] if kw in q] for sid,sp in _CHAT_SPECS.items()}
        hits = {k:v for k,v in hits.items() if v}
        if not hits:
            return ["contract_analyst"]
        ranked = sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)
        top = len(ranked[0][1])
        return [s for s,kws in ranked if len(kws) >= max(1,top//2)]

    for msg in st.session_state["messages"]:
        if msg["role"] == "assistant":
            spec_id = msg.get("specialist", "contract_analyst")
            spec = _CHAT_SPECS.get(spec_id, _CHAT_SPECS["contract_analyst"])
            with st.chat_message("assistant", avatar="🏭"):
                st.markdown(f'<div style="margin-bottom:6px"><span style="background:{spec["color"]}22;color:{spec["color"]};border:1px solid {spec["color"]}55;border-radius:10px;padding:2px 9px;font-size:10px;font-weight:700;letter-spacing:.5px">{spec["icon"]} {spec["name"]}</span></div>', unsafe_allow_html=True)
                st.markdown(msg["content"])
        else:
            with st.chat_message(msg["role"], avatar="👤"):
                st.markdown(msg["content"])

    active_prompt = None
    add_to_history = True
    typed = st.chat_input("Ask about suppliers, contracts, risks, approvals…")
    if typed:
        active_prompt = typed
    elif st.session_state.get("pending_query"):
        active_prompt = st.session_state.pop("pending_query")
        add_to_history = False

    if active_prompt:
        if not api_key:
            st.error("Agent Chat requires a Groq API key. Add GROQ_API_KEY to Streamlit secrets.")
            st.stop()
        if add_to_history:
            st.session_state["messages"].append({"role": "user", "content": active_prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(active_prompt)

        # Show routing decision before LLM responds
        routed_to = _chat_route(active_prompt)
        primary_spec = routed_to[0]
        spec_info = _CHAT_SPECS[primary_spec]
        multi_chat = len(routed_to) > 1
        routing_msg = (f"Routing to **{' + '.join(_CHAT_SPECS[s]['name'] for s in routed_to)}**"
                       if multi_chat else f"Routing to **{spec_info['name']}**")

        with st.chat_message("assistant", avatar="🏭"):
            st.markdown(f'<div style="margin-bottom:8px"><span style="background:{spec_info["color"]}22;color:{spec_info["color"]};border:1px solid {spec_info["color"]}55;border-radius:10px;padding:2px 9px;font-size:10px;font-weight:700;letter-spacing:.5px">{spec_info["icon"]} {spec_info["name"]}{"  +  " + " + ".join(_CHAT_SPECS[s]["name"] for s in routed_to[1:]) if multi_chat else ""}</span>  <span style="font-size:10px;opacity:.4">{routing_msg}</span></div>', unsafe_allow_html=True)
            placeholder = st.empty()
            full, start = "", time.time()
            try:
                from groq import Groq
                client = Groq(api_key=api_key)
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile", max_tokens=1024,
                    messages=[{"role":"system","content":SYSTEM_PROMPT},
                              *[{"role":m["role"],"content":m["content"]} for m in st.session_state["messages"]]],
                    stream=True)
                for chunk in stream:
                    d = chunk.choices[0].delta.content
                    if d:
                        full += d
                        placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                st.caption(f"{(time.time()-start)*1000:.0f}ms · llama-3.3-70b-versatile · routed via supervisor")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.session_state["messages"].append({"role":"assistant","content":full,"specialist":primary_spec})

    if st.session_state.get("messages"):
        if st.button("Clear conversation"):
            st.session_state["messages"] = []; st.rerun()
