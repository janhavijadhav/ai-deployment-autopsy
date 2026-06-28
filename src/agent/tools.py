"""
Agent tools — each wraps a data source call and is registered with LangGraph.

All I/O is async so tools can be parallelised (the fix for Failure 2).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.tools import tool

from src.data.database import get_supplier, search_suppliers, get_contract_metadata
from src.rag.pipeline import retrieve_contracts
from src.auth.oauth2 import SupplierAPIClient
from src.cache.redis_cache import SemanticCache
from src.observability.tracing import tracer


_cache = SemanticCache()
_supplier_client = SupplierAPIClient()


# ─── Contract Search ─────────────────────────────────────────────────────────────

@tool
async def search_contracts(query: str, supplier_id: str | None = None, top_k: int = 5) -> list[dict]:
    """
    Semantic + keyword hybrid search over supplier contracts (PDFs).
    Returns ranked chunks with grounding text.

    Args:
        query:       Natural-language question or clause to find
        supplier_id: Optional — restrict to one supplier's contracts
        top_k:       Number of chunks to return (default 5)
    """
    with tracer.span("tool.search_contracts") as span:
        span.set_attribute("query", query)
        span.set_attribute("supplier_id", supplier_id or "all")

        # Check semantic cache first
        cache_key = f"contracts:{supplier_id}:{query}"
        if cached := await _cache.get(cache_key, query):
            span.set_attribute("cache_hit", True)
            return cached

        results = await retrieve_contracts(query, supplier_id=supplier_id, top_k=top_k)
        await _cache.set(cache_key, query, results)
        return results


# ─── Supplier Lookup ─────────────────────────────────────────────────────────────

@tool
async def lookup_supplier(supplier_id: str) -> dict:
    """
    Fetch a supplier's full record from DuckDB (mirrored SAP data).
    Includes risk score, delivery performance, active contracts.

    Args:
        supplier_id: Internal supplier identifier (e.g. "SUP-0042")
    """
    with tracer.span("tool.lookup_supplier") as span:
        span.set_attribute("supplier_id", supplier_id)
        record = await get_supplier(supplier_id)
        if not record:
            return {"error": f"Supplier {supplier_id} not found", "supplier_id": supplier_id}
        return record


@tool
async def search_suppliers_by_criteria(
    country: str | None = None,
    min_risk_score: float | None = None,
    max_risk_score: float | None = None,
    category: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """
    Search suppliers in the SAP mirror by risk score, country, or category.

    Args:
        country:        ISO country code (e.g. "CN", "DE", "MX")
        min_risk_score: Floor for risk score (0.0 = lowest risk)
        max_risk_score: Ceiling for risk score (1.0 = highest risk)
        category:       Procurement category (e.g. "electronics", "raw_materials")
        limit:          Max results
    """
    with tracer.span("tool.search_suppliers") as span:
        results = await search_suppliers(
            country=country,
            min_risk=min_risk_score,
            max_risk=max_risk_score,
            category=category,
            limit=limit,
        )
        span.set_attribute("result_count", len(results))
        return results


# ─── Live Supplier API (with auth) ──────────────────────────────────────────────

@tool
async def get_live_supplier_status(supplier_id: str) -> dict:
    """
    Fetch real-time supplier status from the external Supplier REST API.
    Uses OAuth2 with automatic token refresh (mutex-protected — Failure 5 fix).

    Args:
        supplier_id: Supplier identifier
    """
    with tracer.span("tool.live_supplier_status") as span:
        span.set_attribute("supplier_id", supplier_id)
        response = await _supplier_client.get(f"/suppliers/{supplier_id}/status")
        return response


# ─── Risk Assessment ─────────────────────────────────────────────────────────────

@tool
async def flag_supplier_risks(supplier_ids: list[str]) -> list[dict]:
    """
    Run risk flagging across multiple suppliers in parallel.
    Returns structured risk flags with severity, category, and description.

    Args:
        supplier_ids: List of supplier IDs to assess
    """
    with tracer.span("tool.flag_risks") as span:
        span.set_attribute("supplier_count", len(supplier_ids))

        # Parallel DB lookups — key part of Failure 2 fix
        records = await asyncio.gather(*[get_supplier(sid) for sid in supplier_ids])

        flags = []
        for record in records:
            if not record:
                continue
            if record["risk_score"] >= 0.7:
                flags.append({
                    "flag_id": f"RISK-{record['supplier_id']}-HIGH",
                    "severity": "critical" if record["risk_score"] >= 0.9 else "high",
                    "category": "risk_score",
                    "description": f"Supplier {record['name']} risk score {record['risk_score']:.2f} exceeds threshold",
                    "supplier_id": record["supplier_id"],
                    "contract_id": None,
                })
            if record["on_time_delivery_pct"] < 85.0:
                flags.append({
                    "flag_id": f"RISK-{record['supplier_id']}-DELIVERY",
                    "severity": "medium",
                    "category": "delivery_delay",
                    "description": f"On-time delivery {record['on_time_delivery_pct']:.1f}% below 85% SLA",
                    "supplier_id": record["supplier_id"],
                    "contract_id": None,
                })
        span.set_attribute("flags_raised", len(flags))
        return flags


# ─── Approval Workflow ───────────────────────────────────────────────────────────

@tool
async def initiate_approval(
    contract_id: str,
    action: str,
    requested_by: str,
    amount_usd: float | None = None,
) -> dict:
    """
    Start a multi-step contract approval workflow.
    Routes to correct approval chain based on dollar threshold.

    Args:
        contract_id:  Contract to approve
        action:       "renew" | "terminate" | "amend"
        requested_by: Username or role requesting approval
        amount_usd:   Contract value for routing thresholds
    """
    with tracer.span("tool.initiate_approval") as span:
        import uuid
        approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"

        # Threshold-based routing
        if amount_usd and amount_usd > 1_000_000:
            chain = ["procurement_manager", "finance_vp", "cpo"]
        elif amount_usd and amount_usd > 100_000:
            chain = ["procurement_manager", "finance_vp"]
        else:
            chain = ["procurement_manager"]

        steps = [
            {
                "step": i + 1,
                "approver_role": role,
                "status": "pending",
                "timestamp": None,
                "notes": None,
            }
            for i, role in enumerate(chain)
        ]

        span.set_attribute("approval_id", approval_id)
        span.set_attribute("approval_steps", len(steps))
        return {
            "approval_id": approval_id,
            "contract_id": contract_id,
            "action": action,
            "requested_by": requested_by,
            "amount_usd": amount_usd,
            "chain": steps,
            "status": "initiated",
        }


# ─── Analytics ──────────────────────────────────────────────────────────────────

@tool
async def get_procurement_analytics(
    metric: str,
    period: str = "last_30_days",
    supplier_id: str | None = None,
) -> dict:
    """
    Query DuckDB for procurement analytics (spend, savings, risk trends).

    Args:
        metric:      "total_spend" | "savings" | "risk_trend" | "contract_expiry_forecast"
        period:      "last_7_days" | "last_30_days" | "last_quarter" | "ytd"
        supplier_id: Optional supplier filter
    """
    with tracer.span("tool.analytics") as span:
        from src.data.database import query_analytics
        span.set_attribute("metric", metric)
        span.set_attribute("period", period)
        result = await query_analytics(metric=metric, period=period, supplier_id=supplier_id)
        return result


# ─── Tool registry ───────────────────────────────────────────────────────────────

ALL_TOOLS = [
    search_contracts,
    lookup_supplier,
    search_suppliers_by_criteria,
    get_live_supplier_status,
    flag_supplier_risks,
    initiate_approval,
    get_procurement_analytics,
]
