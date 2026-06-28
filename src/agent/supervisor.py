"""
Supervisor Agent — classifies incoming queries and delegates to specialist agents.

The supervisor makes routing decisions using a two-tier approach:
1. Fast-path: keyword-based classification (~1ms, no LLM call)
2. Slow-path: LLM-based disambiguation for cross-domain queries (~300ms)

Multi-specialist routing is fully supported: a query like "What are Apex's
penalty clauses and how does their risk score compare?" routes to BOTH
ContractAnalyst AND SupplierRisk. The synthesizer node then merges their outputs
into a single coherent response.

Design rationale
----------------
Having a separate supervisor (rather than a single "do-everything" agent)
gives us three concrete advantages:
  1. Specialists have tighter system prompts → lower hallucination rate
  2. Supervisor routing decisions are auditable (stored in delegation_trace)
  3. Parallel fan-out is explicit in the graph — easy to benchmark vs. serial
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── Specialist registry ────────────────────────────────────────────────────────

SPECIALISTS: dict[str, dict] = {
    "contract_analyst": {
        "name": "Contract Analyst",
        "color": "#4a9eff",
        "icon": "📄",
        "description": "Contract clause search, SLA terms, penalty schedules, renewal dates, payment terms",
        "tools": ["search_contracts", "get_contract_metadata"],
        "keywords": [
            "contract", "clause", "sla", "penalty", "penalt", "terms", "renewal",
            "expire", "expiry", "terminat", "amend", "schedule", "delivery terms",
            "payment terms", "warranty", "liability", "indemnif", "force majeure",
            "net-60", "net-45", "net-30", "invoice", "late", "delay",
        ],
    },
    "supplier_risk": {
        "name": "Supplier Risk",
        "color": "#ff4b4b",
        "icon": "⚠️",
        "description": "Risk scoring, delivery performance, geopolitical exposure, compliance, disputes",
        "tools": ["lookup_supplier", "search_suppliers_by_criteria",
                  "flag_supplier_risks", "get_live_supplier_status"],
        "keywords": [
            "risk", "supplier", "vendor", "manufacturer", "delivery", "performance",
            "on-time", "on time", "probation", "flag", "alert", "exposure",
            "geopolit", "sanction", "certif", "audit", "compliance", "quality",
            "rejection", "dispute", "apex", "brightfield", "dalton", "coretech",
            "pinnacle", "sup-", "tier-",
        ],
    },
    "spend_analytics": {
        "name": "Spend Analytics",
        "color": "#21c354",
        "icon": "📊",
        "description": "Spend analysis, savings identification, budget forecasting, KPI dashboards",
        "tools": ["get_procurement_analytics"],
        "keywords": [
            "spend", "saving", "budget", "forecast", "trend", "analytics",
            "cost", "price variance", "invoice", "quarter", "annual", "q1", "q2",
            "q3", "q4", "ytd", "monthly", "weekly", "dashboard", "kpi", "report",
            "total", "breakdown", "category", "rebate", "discount", "opportunit",
        ],
    },
}


# ── Decision dataclass ─────────────────────────────────────────────────────────

@dataclass
class SupervisorDecision:
    """The supervisor's routing decision for a given query."""
    query: str
    specialists: list[str]           # e.g. ["contract_analyst", "supplier_risk"]
    primary: str                     # which specialist runs first
    reasoning: str                   # human-readable explanation
    confidence: float                # 0.0 → 1.0
    multi_agent: bool                # True if >1 specialist selected
    query_type: str                  # "contract_search" | "risk_assessment" | etc.
    keywords_matched: list[str] = field(default_factory=list)


# ── Core classification logic ──────────────────────────────────────────────────

def classify_query(query: str, llm_classify_fn=None) -> SupervisorDecision:
    """
    Classify a query and decide which specialist(s) should handle it.

    Fast path: keyword matching (O(n) string search, ~1ms).
    Slow path: optional LLM function passed as `llm_classify_fn` for ambiguous cases.

    Parameters
    ----------
    query : str
        The user's natural language query.
    llm_classify_fn : callable | None
        Optional LLM function for disambiguation. Called when multiple specialists
        tie on keyword hits. Signature: (query: str) -> str  (specialist id(s), csv)

    Returns
    -------
    SupervisorDecision
    """
    q = query.lower()

    # Count keyword hits per specialist
    hits: dict[str, list[str]] = {}
    for spec_id, spec in SPECIALISTS.items():
        matched = [kw for kw in spec["keywords"] if kw in q]
        if matched:
            hits[spec_id] = matched

    if not hits:
        # No signal — default to contract_analyst (most common procurement query type)
        return SupervisorDecision(
            query=query,
            specialists=["contract_analyst"],
            primary="contract_analyst",
            reasoning=(
                "No strong domain signal detected. Defaulting to Contract Analyst "
                "as the primary specialist for general procurement queries."
            ),
            confidence=0.45,
            multi_agent=False,
            query_type="general",
            keywords_matched=[],
        )

    # Rank specialists by keyword hit count
    ranked = sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)
    top_count = len(ranked[0][1])

    if len(ranked) == 1:
        # Clear single-domain signal
        spec_id, matched_kw = ranked[0]
        spec = SPECIALISTS[spec_id]
        confidence = min(0.65 + len(matched_kw) * 0.06, 0.97)
        return SupervisorDecision(
            query=query,
            specialists=[spec_id],
            primary=spec_id,
            reasoning=(
                f"Query clearly targets the {spec['name']} domain. "
                f"Matched {len(matched_kw)} domain keyword(s): "
                + ", ".join(f'"{k}"' for k in matched_kw[:4])
                + ". Single-specialist delegation — no synthesis step needed."
            ),
            confidence=confidence,
            multi_agent=False,
            query_type=_infer_query_type(spec_id, q),
            keywords_matched=matched_kw,
        )

    # Multiple specialists have keyword hits
    # Include all that have at least half the top specialist's hit count
    selected = [sid for sid, kws in ranked if len(kws) >= max(1, top_count // 2)]
    primary = ranked[0][0]

    all_kw: list[str] = []
    for _, kws in ranked:
        all_kw.extend(kws)

    spec_names = " + ".join(SPECIALISTS[s]["name"] for s in selected)
    confidence = min(0.70 + len(all_kw) * 0.03, 0.96)

    kw_quoted = ", ".join(f'"{k}"' for k in all_kw[:6])
    reasoning = (
        f"Cross-domain query detected — signals from {len(selected)} specialist domains. "
        f"Delegating to {spec_names} in parallel. "
        f"Keyword hits: {kw_quoted}. "
        f"Outputs will be merged by the synthesizer node."
    )

    return SupervisorDecision(
        query=query,
        specialists=selected,
        primary=primary,
        reasoning=reasoning,
        confidence=confidence,
        multi_agent=len(selected) > 1,
        query_type="cross_domain" if len(selected) > 1 else _infer_query_type(primary, q),
        keywords_matched=all_kw,
    )


def _infer_query_type(specialist: str, query: str) -> str:
    return {
        "contract_analyst": "contract_search",
        "supplier_risk": "risk_assessment",
        "spend_analytics": "spend_analysis",
    }.get(specialist, "general")
