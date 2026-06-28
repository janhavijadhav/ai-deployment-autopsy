"""
Multi-agent LangGraph graph for Meridian's Procurement Intelligence Platform.

Graph topology
--------------

  START
    │
    ▼
  supervisor_node          ← LLM classifies query → picks specialist(s)
    │
    ▼ [conditional routing via Send()]
    ├──► contract_analyst_node ─┐
    ├──► supplier_risk_node    ─┼──► synthesizer_node ──► END
    └──► spend_analytics_node  ─┘

Multi-specialist fan-out
------------------------
When the supervisor selects 2+ specialists, LangGraph's Send() API fans the
same state out to all specialists in parallel.  Each runs independently and
appends its output to `specialist_outputs`.  The synthesizer waits for all
of them (LangGraph handles the join automatically) then merges the outputs.

This demonstrates LangGraph's strength over plain function composition:
parallel execution is handled by the graph runtime, not by hand-written
asyncio.gather() calls in application code.

Checkpointing
-------------
AsyncSqliteSaver checkpoints state after every node, preserving multi-turn
specialist context across API calls — the same Failure 3 fix applied to the
base procurement agent.

State shape
-----------
MultiAgentState extends the conversation fields with:
  supervisor_decision  : routing decision + reasoning (stored for observability)
  specialist_outputs   : {specialist_id: answer_text}
  delegation_trace     : list of step-by-step routing events (for the Failure Museum)
  final_response       : merged output from synthesizer
"""
from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send

from src.agent.supervisor import classify_query, SupervisorDecision, SPECIALISTS
from src.agent.specialists import SpecialistAgent, SpecialistResponse


# ── State ──────────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    # ── Conversation ───────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    turn_count: int
    summary: str | None
    trace_id: str | None

    # ── Routing ────────────────────────────────────────────────────────────────
    supervisor_decision: dict | None   # SupervisorDecision serialised to dict
    pending_specialists: list[str]     # specialists remaining in this turn
    active_specialist: str | None      # specialist currently executing

    # ── Outputs ────────────────────────────────────────────────────────────────
    specialist_outputs: dict[str, str] # specialist_id → answer text
    delegation_trace: list[dict]       # step-by-step routing events

    # ── Final ──────────────────────────────────────────────────────────────────
    final_response: str | None
    latency_ms: float | None


# ── Nodes ──────────────────────────────────────────────────────────────────────

async def supervisor_node(state: MultiAgentState) -> dict:
    """
    Supervisor: classify the incoming query and decide which specialist(s) to route to.

    Stores the full SupervisorDecision in state under `supervisor_decision` so the
    delegation_trace is complete for observability and the Failure Museum demo.
    """
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    decision: SupervisorDecision = classify_query(query)

    trace_entry = {
        "step": "supervisor",
        "node": "supervisor",
        "decision": decision.specialists,
        "primary": decision.primary,
        "reasoning": decision.reasoning,
        "confidence": round(decision.confidence, 3),
        "multi_agent": decision.multi_agent,
        "keywords_matched": decision.keywords_matched[:6],
    }

    return {
        "supervisor_decision": {
            "query": decision.query,
            "specialists": decision.specialists,
            "primary": decision.primary,
            "reasoning": decision.reasoning,
            "confidence": decision.confidence,
            "multi_agent": decision.multi_agent,
            "query_type": decision.query_type,
            "keywords_matched": decision.keywords_matched,
        },
        "pending_specialists": decision.specialists,
        "active_specialist": None,
        "specialist_outputs": {},
        "delegation_trace": [trace_entry],
        "trace_id": str(uuid.uuid4()),
    }


async def _run_specialist(state: MultiAgentState, specialist_id: str) -> dict:
    """
    Generic specialist runner — called by each specialist node.

    Appends to specialist_outputs and delegation_trace without overwriting
    outputs from other specialists running in parallel.
    """
    last_msg = state["messages"][-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    agent = SpecialistAgent(specialist_id)
    result: SpecialistResponse = await agent.respond(query)

    # Merge into existing outputs (parallel specialists each write their own key)
    existing_outputs = dict(state.get("specialist_outputs") or {})
    existing_outputs[specialist_id] = result.answer

    existing_trace = list(state.get("delegation_trace") or [])
    existing_trace.append({
        "step": specialist_id,
        "node": f"{specialist_id}_node",
        "specialist_name": result.specialist_name,
        "answer_preview": result.answer[:100] + "…",
        "sources": result.sources,
        "tool_calls": result.tool_calls,
        "latency_ms": round(result.latency_ms, 1),
        "confidence": result.confidence,
    })

    return {
        "specialist_outputs": existing_outputs,
        "active_specialist": specialist_id,
        "delegation_trace": existing_trace,
    }


async def contract_analyst_node(state: MultiAgentState) -> dict:
    return await _run_specialist(state, "contract_analyst")


async def supplier_risk_node(state: MultiAgentState) -> dict:
    return await _run_specialist(state, "supplier_risk")


async def spend_analytics_node(state: MultiAgentState) -> dict:
    return await _run_specialist(state, "spend_analytics")


async def synthesizer_node(state: MultiAgentState) -> dict:
    """
    Synthesizer: merge outputs from one or more specialists into a final response.

    For single-specialist queries: passes the answer through unchanged.
    For multi-specialist queries: adds a cross-domain header and merges sections.

    The synthesizer is always the last node before END, even for single-specialist
    routing — this keeps the graph topology uniform and makes the trace complete.
    """
    outputs = state.get("specialist_outputs") or {}

    if not outputs:
        return {
            "final_response": "No specialist outputs were produced.",
            "messages": [AIMessage(content="No specialist outputs were produced.")],
        }

    if len(outputs) == 1:
        answer = next(iter(outputs.values()))
        return {
            "final_response": answer,
            "messages": [AIMessage(content=answer)],
            "turn_count": state.get("turn_count", 0) + 1,
        }

    # ── Multi-specialist synthesis ─────────────────────────────────────────────
    specialist_names = [
        SPECIALISTS[sid]["name"] for sid in outputs if sid in SPECIALISTS
    ]
    header = (
        f"*Cross-domain analysis — {len(outputs)} specialists consulted in parallel: "
        f"{', '.join(specialist_names)}*\n\n"
    )

    sections = []
    for spec_id, answer in outputs.items():
        spec_name = SPECIALISTS.get(spec_id, {}).get("name", spec_id)
        sections.append(f"### {spec_name}\n\n{answer}")

    final = header + "\n\n---\n\n".join(sections)

    existing_trace = list(state.get("delegation_trace") or [])
    existing_trace.append({
        "step": "synthesizer",
        "node": "synthesizer",
        "specialists_merged": list(outputs.keys()),
        "output_chars": len(final),
    })

    return {
        "final_response": final,
        "messages": [AIMessage(content=final)],
        "turn_count": state.get("turn_count", 0) + 1,
        "delegation_trace": existing_trace,
    }


# ── Routing functions ──────────────────────────────────────────────────────────

def route_after_supervisor(state: MultiAgentState) -> list[Send]:
    """
    Conditional edge after supervisor_node.

    Returns a list of Send objects — one per selected specialist.
    LangGraph executes all of them in parallel and joins their state
    updates before passing control to the synthesizer.
    """
    decision = state.get("supervisor_decision") or {}
    specialists = decision.get("specialists", ["contract_analyst"])

    node_map = {
        "contract_analyst": "contract_analyst_node",
        "supplier_risk": "supplier_risk_node",
        "spend_analytics": "spend_analytics_node",
    }

    return [Send(node_map[spec], state) for spec in specialists if spec in node_map]


# ── Build graph ────────────────────────────────────────────────────────────────

def build_multi_agent_graph(checkpointer=None):
    """
    Compile the multi-agent LangGraph.

    Supports:
    - Single-specialist routing (most queries — lowest latency)
    - Multi-specialist fan-out via Send() (parallel execution, auto-joined)
    - Optional SQLite checkpointing for multi-turn state persistence

    Parameters
    ----------
    checkpointer : LangGraph checkpointer | None
        Use AsyncSqliteSaver in production. Pass None for stateless/testing.
    """
    graph = StateGraph(MultiAgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("contract_analyst_node", contract_analyst_node)
    graph.add_node("supplier_risk_node", supplier_risk_node)
    graph.add_node("spend_analytics_node", spend_analytics_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Entry ──────────────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # ── Fan-out: supervisor → parallel specialist(s) ───────────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        [
            "contract_analyst_node",
            "supplier_risk_node",
            "spend_analytics_node",
        ],
    )

    # ── Convergence: all specialists → synthesizer ──────────────────────────────
    graph.add_edge("contract_analyst_node", "synthesizer")
    graph.add_edge("supplier_risk_node", "synthesizer")
    graph.add_edge("spend_analytics_node", "synthesizer")

    # ── Exit ───────────────────────────────────────────────────────────────────
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


# ── Public interface ───────────────────────────────────────────────────────────

async def run_multi_agent(
    query: str,
    thread_id: str = "default",
    checkpointer=None,
) -> dict:
    """
    Run the multi-agent graph for one query and return the full final state.

    Returned dict includes:
      final_response     : str  — merged answer
      delegation_trace   : list — step-by-step routing log
      specialist_outputs : dict — per-specialist answer text
      supervisor_decision: dict — routing decision + reasoning

    Parameters
    ----------
    query       : User's natural language question.
    thread_id   : Conversation thread ID for checkpointing.
    checkpointer: Optional LangGraph checkpointer for persistence.
    """
    graph = build_multi_agent_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: MultiAgentState = {
        "messages": [HumanMessage(content=query)],
        "turn_count": 0,
        "summary": None,
        "trace_id": None,
        "supervisor_decision": None,
        "pending_specialists": [],
        "active_specialist": None,
        "specialist_outputs": {},
        "delegation_trace": [],
        "final_response": None,
        "latency_ms": None,
    }

    return await graph.ainvoke(initial_state, config=config)
