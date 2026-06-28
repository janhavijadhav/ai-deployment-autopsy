"""
Specialist Agents — domain-focused agents with narrow system prompts and targeted tool sets.

Each specialist:
  - Has a system prompt scoped exclusively to its domain
  - Only has access to tools relevant to its domain (principle of least privilege)
  - Returns a SpecialistResponse: structured answer + sources + confidence + latency
  - Is async-safe — can run in parallel with other specialists via asyncio.gather

Design principle
----------------
A narrow specialist beats a single generalist for three reasons:
  1. The system prompt doesn't context-switch between domains → fewer hallucinations
  2. Tool selection is tighter → LLM can't accidentally call a spend analytics tool
     when answering a contract question
  3. Independent confidence scoring per domain enables the synthesizer to weight outputs

Simulation mode
---------------
For the Failure Museum Streamlit demo, specialists use deterministic simulation
(no API call). The simulation returns realistic-looking responses based on keyword
matching — good enough to demonstrate the delegation pattern without spending API quota.
Real LLM mode is activated by passing `llm=<ChatModel>` to respond().
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.agent.supervisor import SPECIALISTS


# ── Response type ─────────────────────────────────────────────────────────────

@dataclass
class SpecialistResponse:
    """Structured output from one specialist agent."""
    specialist_id: str
    specialist_name: str
    answer: str
    sources: list[str]       # contract IDs, supplier IDs, analytics refs cited
    confidence: float        # 0.0 → 1.0
    tool_calls: list[str]    # which tools were invoked
    latency_ms: float
    metadata: dict[str, Any]


# ── System prompts ────────────────────────────────────────────────────────────

CONTRACT_ANALYST_PROMPT = """You are the Contract Analyst specialist for Meridian Manufacturing Corp.

SCOPE (strict): Contract clause search, SLA terms, penalty schedules, renewal dates,
payment terms, warranty conditions, IP rights, termination clauses, force majeure.

GROUNDING RULES:
- Only cite clauses from retrieved RAG chunks. Never invent contract terms or clause numbers.
- Always cite: contract_id + clause/section reference (e.g. "CTR-00001 §8.2").
- If the query touches supplier risk or spend analysis, acknowledge it and say
  "Supplier Risk Analyst has this domain" — do NOT attempt to answer outside your scope.
- Precision over recall: one accurate citation beats five uncertain ones.

Output structure:
1. Direct answer with exact clause reference
2. Adjacent context (neighbouring clauses if relevant)
3. Risk note if the clause has unusual or non-standard terms
"""

SUPPLIER_RISK_PROMPT = """You are the Supplier Risk Analyst specialist for Meridian Manufacturing Corp.

SCOPE (strict): Supplier risk scoring (0.0–1.0 scale), delivery performance, quality
rejection rates, geopolitical exposure, certification status, open disputes, probation status.

GROUNDING RULES:
- Risk score ≥ 0.70 → flag as HIGH, require escalation to CPO within 48 hours.
- Risk score ≥ 0.90 → flag as CRITICAL, immediate action required.
- On-time delivery < 85% triggers automatic SLA breach notification.
- Always cite supplier_id (e.g. SUP-0001) and data source.
- Cross-reference country-level geopolitical risk when relevant.

Output structure:
1. Risk score and tier (LOW / MEDIUM / HIGH / CRITICAL)
2. Top risk factors with supporting metrics
3. Recommended action: MONITOR / ESCALATE / SUSPEND_POS / TERMINATE
"""

SPEND_ANALYTICS_PROMPT = """You are the Spend Analytics specialist for Meridian Manufacturing Corp.

SCOPE (strict): Total portfolio spend ($2.4B annually), category spend breakdown,
savings opportunities (volume rebates, early payment discounts, consolidation),
budget vs. actual, price trend analysis, contract value forecasting.

GROUNDING RULES:
- All figures in USD unless stated otherwise.
- Always include a time period with any metric (e.g. "Q3 YTD", "last 30 days").
- Identify concrete savings opportunities with dollar amounts.
- If a query touches contract clauses or supplier risk, flag it and say
  which specialist owns that domain.

Output structure:
1. Key metric with period and trend direction
2. Top driver or insight
3. Actionable recommendation with estimated $ impact
"""


# ── Specialist agent ──────────────────────────────────────────────────────────

class SpecialistAgent:
    """
    A domain-focused LLM agent scoped to a specific procurement subdomain.

    Production mode: calls an LLM with a narrow system prompt and domain-specific tools.
    Demo mode: uses deterministic simulation (no API call, instant response).
    """

    def __init__(self, specialist_id: str):
        if specialist_id not in SPECIALISTS:
            raise ValueError(
                f"Unknown specialist '{specialist_id}'. Valid: {list(SPECIALISTS)}"
            )
        self.id = specialist_id
        self.meta = SPECIALISTS[specialist_id]
        self.name: str = self.meta["name"]
        self.color: str = self.meta["color"]
        self.icon: str = self.meta["icon"]
        self.tool_names: list[str] = self.meta["tools"]

    async def respond(
        self,
        query: str,
        llm=None,
        context: dict | None = None,
    ) -> SpecialistResponse:
        """
        Process a query and return a structured response.

        Parameters
        ----------
        query   : The user's question.
        llm     : Optional LangChain chat model. If None, uses deterministic simulation.
        context : Optional extra context dict (retrieved chunks, state, etc.)
        """
        t0 = time.perf_counter()

        if llm is not None:
            answer, sources, tool_calls = await self._respond_llm(query, llm, context or {})
            confidence = 0.91
        else:
            answer, sources, tool_calls = self._respond_simulated(query, context or {})
            confidence = 0.83

        latency = (time.perf_counter() - t0) * 1000

        return SpecialistResponse(
            specialist_id=self.id,
            specialist_name=self.name,
            answer=answer,
            sources=sources,
            confidence=confidence,
            tool_calls=tool_calls,
            latency_ms=latency,
            metadata={"query": query, "simulated": llm is None},
        )

    async def _respond_llm(
        self, query: str, llm, context: dict
    ) -> tuple[str, list[str], list[str]]:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompts = {
            "contract_analyst": CONTRACT_ANALYST_PROMPT,
            "supplier_risk": SUPPLIER_RISK_PROMPT,
            "spend_analytics": SPEND_ANALYTICS_PROMPT,
        }
        messages = [
            SystemMessage(content=prompts[self.id]),
            HumanMessage(content=query),
        ]
        response = await llm.ainvoke(messages)
        return response.content, [], self.tool_names[:2]

    def _respond_simulated(
        self, query: str, context: dict
    ) -> tuple[str, list[str], list[str]]:
        q = query.lower()
        dispatch = {
            "contract_analyst": _simulate_contract_analyst,
            "supplier_risk": _simulate_supplier_risk,
            "spend_analytics": _simulate_spend_analytics,
        }
        return dispatch[self.id](q)


# ── Simulation helpers ────────────────────────────────────────────────────────

def _simulate_contract_analyst(q: str) -> tuple[str, list[str], list[str]]:
    if any(w in q for w in ["penalt", "delay", "late delivery", "late fee"]):
        return (
            "**Contract Analyst — Penalty Clause Analysis**\n\n"
            "Source: CTR-00001 (Apex Industries), Section 8.2 — Delivery Penalties\n\n"
            "| Delay Window | Penalty Rate |\n"
            "|---|---|\n"
            "| Days 1–30 | 0.5% of invoice value per calendar week |\n"
            "| Days 31–60 | 1.5% per week (compounded weekly) |\n"
            "| Day 61+ | 3.0% per week + right to terminate immediately |\n\n"
            "**Risk Note:** The Day 61+ clause includes consequential loss recovery "
            "(§8.2.4) — non-standard for Tier-2 suppliers. The compound interest "
            "mechanism on Days 31–60 was flagged during contract review but accepted. "
            "Recommend legal review before the next renewal cycle.",
            ["CTR-00001 §8.2", "CTR-00001 §8.2.4", "chunk-day61-plus"],
            ["search_contracts"],
        )
    elif any(w in q for w in ["expir", "renew", "terminat"]):
        return (
            "**Contract Analyst — Expiry & Renewal Report**\n\n"
            "Active contract timeline:\n\n"
            "• **CTR-00001** (Apex Industries) — Expires 2024-12-31\n"
            "  Auto-renewal clause §14.1 active. 120-day notice window opens 2024-09-01. "
            "  **Action required by Sep 1.**\n\n"
            "• **CTR-00003** (Dalton Materials) — Expires 2024-06-30 ⚠️\n"
            "  Supplier is on probation. Renewal is NOT recommended pending Q3 audit.\n\n"
            "• **CTR-00002** (Brightfield Components) — Expires 2025-03-31\n"
            "  Normal renewal cycle. Performance satisfactory.\n\n"
            "Immediate action: Initiate review for CTR-00003 (Dalton). "
            "No auto-renewal — decision required.",
            ["CTR-00001 §14.1", "CTR-00002 §14.1", "CTR-00003 §14.1"],
            ["search_contracts", "get_contract_metadata"],
        )
    elif any(w in q for w in ["payment", "net-", "net ", "invoice"]):
        return (
            "**Contract Analyst — Payment Terms Summary**\n\n"
            "| Supplier | Payment Terms | Early Pay Discount |\n"
            "|---|---|---|\n"
            "| Apex Industries (CTR-00001) | Net-60 | 2% if paid Net-10 |\n"
            "| Brightfield (CTR-00002) | Net-45 | None |\n"
            "| Dalton Materials (CTR-00003) | Net-30 | None (tightened from Net-45) |\n\n"
            "**Savings Insight:** Capturing the Apex early-pay discount on $28.4M "
            "annual spend = **$568K in annual savings** (§9.1.3). "
            "Currently uncaptured due to AP batch processing delays.",
            ["CTR-00001 §9.1", "CTR-00002 §9.1", "CTR-00003 §9.1"],
            ["search_contracts"],
        )
    else:
        return (
            "**Contract Analyst — General Clause Search**\n\n"
            "Retrieved 3 relevant clauses across 2 active agreements:\n\n"
            "• **CTR-00001 §4.3** — Force majeure: covers supply chain disruption "
            "exceeding 14 consecutive days. Supplier must notify within 48h.\n\n"
            "• **CTR-00002 §7.1** — SLA uptime: 99.5% monthly availability required. "
            "Breach triggers service credit equal to 10% of monthly invoice.\n\n"
            "• **CTR-00001 §11.2** — IP ownership: custom tooling and dies revert to "
            "Meridian Manufacturing upon contract expiry or termination.\n\n"
            "All clauses cross-referenced against Meridian Standard Template v3.2. "
            "No non-standard deviations detected in this search.",
            ["CTR-00001 §4.3", "CTR-00002 §7.1", "CTR-00001 §11.2"],
            ["search_contracts"],
        )


def _simulate_supplier_risk(q: str) -> tuple[str, list[str], list[str]]:
    if any(w in q for w in ["apex", "sup-0001"]):
        return (
            "**Supplier Risk — Apex Industries (SUP-0001)**\n\n"
            "Overall Risk Score: **0.82 / 1.0** 🔴 CRITICAL\n\n"
            "**Top risk factors:**\n\n"
            "1. **Delivery Performance** (weight: 35%)\n"
            "   On-time rate: 87% vs. SLA of 95%. Third consecutive quarter below threshold. "
            "   Trend: declining (Q1: 91%, Q2: 89%, Q3: 87%).\n\n"
            "2. **Geopolitical Exposure** (weight: 30%)\n"
            "   Headquarters: CN. Active Section 301 tariff review on electronics components. "
            "   Alternative sourcing lead time: 6–9 months.\n\n"
            "3. **Financial Health** (weight: 25%)\n"
            "   Q2 2024 revenue declined 12% YoY. Debt/equity ratio elevated at 2.3× "
            "(threshold: 1.5×). Credit watch: Moody's negative outlook.\n\n"
            "**Recommended Action: ESCALATE**\n"
            "Activate dual-sourcing contingency plan immediately. "
            "Brief CPO within 48 hours per Risk Protocol §3.2.",
            ["SUP-0001", "RISK-SUP-0001-HIGH", "RISK-SUP-0001-DELIVERY"],
            ["lookup_supplier", "flag_supplier_risks", "get_live_supplier_status"],
        )
    elif any(w in q for w in ["probation", "dalton", "sup-0003"]):
        return (
            "**Supplier Risk — Dalton Materials (SUP-0003)**\n\n"
            "Overall Risk Score: **0.67 / 1.0** 🟡 HIGH\n"
            "Status: **PROBATION** (since 2024-02-15)\n\n"
            "**Risk factors:**\n\n"
            "1. **Quality Rejection Rate**: 9.0% (threshold: 2.0%) — 4.5× above limit\n"
            "   Root cause: inadequate QC at Monterrey facility. On-site audit scheduled.\n\n"
            "2. **Open Disputes**: 4 active ($1.2M combined). Legal review in progress.\n\n"
            "3. **Delivery**: On-time rate 71% — lowest in active portfolio.\n\n"
            "**Recommended Action: SUSPEND NEW POs**\n"
            "Hold all new purchase orders pending Q3 quality audit results (due Sep 30).",
            ["SUP-0003", "RISK-SUP-0003-QUALITY", "RISK-SUP-0003-DELIVERY"],
            ["lookup_supplier", "flag_supplier_risks"],
        )
    elif any(w in q for w in ["high risk", "risk score", "above 0.7", "critical"]):
        return (
            "**Supplier Risk — High-Risk Portfolio Scan**\n\n"
            "Suppliers above 0.70 risk threshold: **2 of 5 active**\n\n"
            "| Supplier | Risk Score | Tier | Primary Flag |\n"
            "|---|---|---|---|\n"
            "| Apex Industries (SUP-0001) | 0.82 | 🔴 CRITICAL | CN geopolitical + delivery |\n"
            "| Dalton Materials (SUP-0003) | 0.67 | 🟡 HIGH | Quality failures + disputes |\n\n"
            "Combined at-risk spend: **$37.5M** (Apex $28.4M + Dalton $9.1M)\n\n"
            "**Safe suppliers:**\n"
            "• Brightfield (SUP-0002): 0.21 ✅ | CoreTech (SUP-0004): 0.33 ✅ "
            "| Pinnacle (SUP-0005): 0.44 ✅",
            ["SUP-0001", "SUP-0003"],
            ["search_suppliers_by_criteria", "flag_supplier_risks"],
        )
    else:
        return (
            "**Supplier Risk — Portfolio Overview**\n\n"
            "Active suppliers: 5 | At-risk (≥0.70): 2 | On probation: 1\n\n"
            "| Supplier | Risk Score | Tier | On-Time % |\n"
            "|---|---|---|---|\n"
            "| Apex Industries | 0.82 | 🔴 CRITICAL | 87% |\n"
            "| Dalton Materials | 0.67 | 🟡 HIGH | 71% |\n"
            "| Pinnacle Logistics | 0.44 | 🟡 MODERATE | 89% |\n"
            "| CoreTech Systems | 0.33 | 🟢 LOW | 93% |\n"
            "| Brightfield Components | 0.21 | 🟢 LOW | 97% |\n\n"
            "Portfolio weighted average risk: **0.49** (above 0.40 internal benchmark)\n"
            "Q3 trend: risk score increased +0.06pp vs. Q2 (Apex deterioration).",
            ["SUP-0001", "SUP-0002", "SUP-0003", "SUP-0004", "SUP-0005"],
            ["search_suppliers_by_criteria", "flag_supplier_risks"],
        )


def _simulate_spend_analytics(q: str) -> tuple[str, list[str], list[str]]:
    if any(w in q for w in ["saving", "opportunit", "cost reduc", "discount", "rebate"]):
        return (
            "**Spend Analytics — Savings Opportunity Report**\n\n"
            "Total addressable savings identified: **$4.2M** (1.75% of annual spend)\n\n"
            "**Top 4 opportunities:**\n\n"
            "1. **Early Payment Discount Capture** — $568K\n"
            "   Apex Industries: Net-60 → 2% discount if paid Net-10. "
            "Currently uncaptured on $28.4M.\n\n"
            "2. **Volume Consolidation — Electronics** — $1.8M\n"
            "   Fragmented spend across 3 suppliers. Single-source to Brightfield "
            "at scale saves 7% on electronics category.\n\n"
            "3. **Pinnacle Rebate Trigger** — $890K\n"
            "   $30M threshold triggers 2.8% rebate. YTD spend: $28.7M. "
            "Only $1.3M away — accelerate POs to unlock.\n\n"
            "4. **Dalton Replacement Net Savings** — $960K\n"
            "   Quality rework costs $960K YTD. Switching to certified alternate "
            "saves net $640K after transition costs.",
            ["SAVINGS-Q3-2024", "SPEND-ANALYSIS"],
            ["get_procurement_analytics"],
        )
    elif any(w in q for w in ["total spend", "budget", "ytd", "year to date", "annual"]):
        return (
            "**Spend Analytics — YTD 2024 Dashboard**\n\n"
            "Total portfolio spend: **$1.82B** (76% of $2.4B annual budget)\n\n"
            "**By supplier (top 5):**\n\n"
            "| Supplier | YTD Spend | % Portfolio | YoY |\n"
            "|---|---|---|---|\n"
            "| Brightfield Components | $41.2M | 22.6% | ↑ 8% |\n"
            "| Pinnacle Logistics | $28.7M | 15.8% | ↑ 3% |\n"
            "| Apex Industries | $23.1M | 12.7% | ↓ 14% |\n"
            "| CoreTech Systems | $15.2M | 8.4% | → 0% |\n"
            "| Dalton Materials | $6.9M | 3.8% | ↓ 24% |\n\n"
            "Budget variance: **−$180M** vs. plan (Apex supply disruption + Dalton PO freeze)\n"
            "Full-year forecast: **$2.35B** — 2% under budget.",
            ["SPEND-YTD-2024", "BUDGET-VARIANCE"],
            ["get_procurement_analytics"],
        )
    elif any(w in q for w in ["forecast", "q4", "next quarter", "trend"]):
        return (
            "**Spend Analytics — Q4 2024 Forecast**\n\n"
            "Projected Q4 spend: **$580M** (vs. $595M Q4 2023, −2.5%)\n\n"
            "**Key forecast drivers:**\n\n"
            "• Apex ramp-down: −$8M vs. Q4 2023 (contingency sourcing active)\n"
            "• Brightfield accelerated production: +$6M\n"
            "• Dalton suspension: −$3M (POs on hold pending quality audit)\n\n"
            "**Risk to forecast:**\n"
            "Apex contingency sourcing may add **$12M in spot market premiums** "
            "if dual-source is not activated by November 1. "
            "This would push Q4 over budget by 2.1%.\n\n"
            "Contract renewals due Q4: 3 agreements ($47M combined value).",
            ["FORECAST-Q4-2024", "RENEWAL-PIPELINE"],
            ["get_procurement_analytics"],
        )
    else:
        return (
            "**Spend Analytics — Last 30 Days KPI Summary**\n\n"
            "| KPI | Value | Trend |\n"
            "|---|---|---|\n"
            "| Total spend | $152M | ↑ 4% MoM |\n"
            "| POs issued | 1,247 | → flat |\n"
            "| Avg PO value | $121K | ↑ 3% |\n"
            "| Contract coverage | 94.2% | ↑ 2.3pp |\n"
            "| Savings realised | $2.1M (1.38%) | ↑ 0.4pp |\n"
            "| On-time payments | 96.4% | ↑ 1.1pp |\n\n"
            "**Insight:** Contract coverage improving — spot buy reduction "
            "driven by Brightfield volume consolidation initiative. "
            "On track to hit 95% coverage target by year-end.",
            ["KPI-LAST-30-DAYS"],
            ["get_procurement_analytics"],
        )


# ── Convenience functions ─────────────────────────────────────────────────────

def get_specialist(specialist_id: str) -> SpecialistAgent:
    """Return a SpecialistAgent for the given ID."""
    return SpecialistAgent(specialist_id)


async def run_specialists_parallel(
    specialists: list[str],
    query: str,
    llm=None,
    context: dict | None = None,
) -> list[SpecialistResponse]:
    """
    Run multiple specialists concurrently using asyncio.gather.

    This is the fan-out pattern used by the multi-agent graph:
    all selected specialists start simultaneously, reducing total latency
    to max(individual_latencies) rather than sum(individual_latencies).
    """
    agents = [SpecialistAgent(s) for s in specialists]
    return await asyncio.gather(
        *[a.respond(query, llm=llm, context=context) for a in agents]
    )
