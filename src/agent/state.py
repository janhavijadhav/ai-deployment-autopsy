"""
LangGraph state for the Procurement Intelligence Agent.

State is persisted via SQLite checkpointing (the fix for Failure 3).
Every field must be JSON-serialisable so LangGraph can checkpoint it.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ─── Sub-types ──────────────────────────────────────────────────────────────────

class SupplierRecord(TypedDict):
    supplier_id: str
    name: str
    country: str
    risk_score: float          # 0.0 (low) → 1.0 (critical)
    active_contracts: int
    on_time_delivery_pct: float
    last_audit_date: str


class ContractChunk(TypedDict):
    chunk_id: str
    contract_id: str
    supplier_id: str
    content: str
    score: float               # retrieval relevance
    metadata: dict[str, Any]


class ApprovalStep(TypedDict):
    step: int
    approver_role: str
    status: Literal["pending", "approved", "rejected", "escalated"]
    timestamp: str | None
    notes: str | None


class RiskFlag(TypedDict):
    flag_id: str
    severity: Literal["low", "medium", "high", "critical"]
    category: str              # e.g. "delivery_delay", "contract_expiry", "price_variance"
    description: str
    supplier_id: str
    contract_id: str | None


# ─── Main Agent State ────────────────────────────────────────────────────────────

class ProcurementState(TypedDict):
    # Conversation — append-only via LangGraph's add_messages reducer
    messages: Annotated[list, add_messages]

    # Query routing
    query_intent: Literal[
        "contract_search",
        "supplier_lookup",
        "risk_assessment",
        "approval_workflow",
        "analytics",
        "unknown",
    ] | None

    # Retrieved data
    supplier_records: list[SupplierRecord]
    contract_chunks: list[ContractChunk]
    risk_flags: list[RiskFlag]

    # Approval workflow (multi-step — persisted across turns)
    approval_chain: list[ApprovalStep]
    approval_id: str | None
    approval_status: Literal["initiated", "in_progress", "approved", "rejected"] | None

    # Context management
    turn_count: int
    summary: str | None        # compressed history summary after Failure 3 fix

    # Observability
    trace_id: str | None
    latency_ms: float | None
    tool_calls_made: list[str]

    # Error handling
    last_error: str | None
    retry_count: int
