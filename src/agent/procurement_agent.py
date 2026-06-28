"""
Multi-agent Procurement Intelligence Platform — LangGraph orchestrator.

Graph topology:
  START → route_query → [contract_searcher | supplier_fetcher | risk_assessor
                          | approval_router | analytics_node] → synthesize → END

Failure 3 fix: SQLite-backed checkpointer preserves state across multi-turn
approval workflows. Without it, the agent loses context after step 3 and
restarts the entire conversation.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Literal

from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.state import ProcurementState
from src.agent.tools import ALL_TOOLS
from src.observability.tracing import tracer, metrics
from src.config import settings

# ─── System Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Procurement Intelligence Agent for Meridian Manufacturing Corp,
a Fortune 500 industrial manufacturer with 1,200+ active supplier contracts worth $2.4B annually.

Your role is to help procurement analysts, finance teams, and operations managers by:
1. Answering questions about supplier contracts (compliance, pricing, SLA terms)
2. Surfacing supplier risk signals (delivery performance, financial health, geopolitical exposure)
3. Routing contract approval workflows through the correct approval chain
4. Providing spend analytics and savings opportunity identification

CRITICAL GROUNDING RULES:
- ONLY answer based on retrieved contract text and SAP data. Never invent supplier names,
  contract terms, prices, or supplier IDs. If information is not in retrieved context, say so.
- When quoting contract terms, cite the exact contract_id and chunk_id.
- Risk scores are on a 0.0–1.0 scale. Flag anything ≥ 0.7 as requiring immediate review.

Current operator: {operator_role}
Approval authority level: {approval_level}
"""


# ─── Graph Nodes ──────────────────────────────────────────────────────────────

async def route_query(state: ProcurementState) -> ProcurementState:
    """
    Classifies the user's intent and sets query_intent.
    This determines which downstream tool nodes execute.
    """
    with tracer.span("node.route_query") as span:
        last_msg = state["messages"][-1]
        query_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # Simple keyword routing — in prod, use a classifier
        query_lower = query_text.lower()
        if any(w in query_lower for w in ["contract", "clause", "terms", "sla", "penalty"]):
            intent = "contract_search"
        elif any(w in query_lower for w in ["supplier", "vendor", "manufacturer"]):
            intent = "supplier_lookup"
        elif any(w in query_lower for w in ["risk", "flag", "alert", "exposure"]):
            intent = "risk_assessment"
        elif any(w in query_lower for w in ["approve", "approval", "sign off", "authorize"]):
            intent = "approval_workflow"
        elif any(w in query_lower for w in ["spend", "savings", "analytics", "forecast", "trend"]):
            intent = "analytics"
        else:
            intent = "unknown"

        span.set_attribute("intent", intent)
        span.set_attribute("turn", state.get("turn_count", 0))
        return {**state, "query_intent": intent, "trace_id": str(uuid.uuid4())}


async def summarize_if_needed(state: ProcurementState) -> ProcurementState:
    """
    Failure 3 fix: Compress old turns into a summary when approaching context limits.
    Preserves full fidelity of recent turns while preventing context window overflow.

    Without this: agent truncates raw messages, losing approval workflow state after
    step 3 and restarting the conversation from scratch.
    """
    messages = state.get("messages", [])
    turn_count = state.get("turn_count", 0)

    # Only summarise if we have many turns and no recent summary
    SUMMARY_EVERY_N_TURNS = 8
    if turn_count > 0 and turn_count % SUMMARY_EVERY_N_TURNS == 0:
        with tracer.span("node.summarize_context") as span:
            llm = ChatAnthropic(model=settings.CLAUDE_MODEL, temperature=0)

            # Keep last 4 turns verbatim; summarise everything before
            recent_messages = messages[-8:]
            old_messages = messages[:-8]

            if not old_messages:
                return state

            summary_prompt = f"""Summarise the following procurement conversation history concisely.
Preserve: all supplier IDs, contract IDs, approval IDs, risk flags, and approval chain status.
Discard: pleasantries, repetition, tool call internals.

History to summarise:
{chr(10).join(str(m) for m in old_messages)}
"""
            response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
            summary_text = response.content

            span.set_attribute("turns_summarised", len(old_messages))
            span.set_attribute("summary_tokens", len(summary_text.split()))

            # Replace old messages with summary message
            summary_msg = SystemMessage(
                content=f"[CONVERSATION SUMMARY — turns 1-{turn_count - 4}]\n{summary_text}"
            )
            new_messages = [summary_msg] + recent_messages

            return {**state, "messages": new_messages, "summary": summary_text}

    return state


async def call_llm_with_tools(state: ProcurementState) -> ProcurementState:
    """Main LLM node — calls Claude with tools bound."""
    with tracer.span("node.llm") as span:
        t0 = time.perf_counter()

        llm = ChatAnthropic(model=settings.CLAUDE_MODEL, temperature=0.1)
        llm_with_tools = llm.bind_tools(ALL_TOOLS)

        system_msg = SystemMessage(
            content=SYSTEM_PROMPT.format(
                operator_role="Procurement Analyst",
                approval_level="Manager",
            )
        )
        messages = [system_msg] + state["messages"]

        response = await llm_with_tools.ainvoke(messages)

        latency = (time.perf_counter() - t0) * 1000
        span.set_attribute("latency_ms", latency)
        metrics.record_llm_latency(latency)

        return {
            **state,
            "messages": [response],
            "turn_count": state.get("turn_count", 0) + 1,
            "latency_ms": latency,
        }


def should_continue(state: ProcurementState) -> Literal["tools", "__end__"]:
    """Route to tools if the LLM made tool calls, else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


# ─── Build the Graph ──────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Builds and compiles the LangGraph procurement agent.

    Args:
        checkpointer: LangGraph checkpointer (AsyncSqliteSaver in prod,
                      None for stateless/testing). Without a checkpointer,
                      multi-step approval workflows lose state between turns.
    """
    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(ProcurementState)

    # Nodes
    graph.add_node("route_query", route_query)
    graph.add_node("summarize_if_needed", summarize_if_needed)
    graph.add_node("agent", call_llm_with_tools)
    graph.add_node("tools", tool_node)

    # Edges
    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "summarize_if_needed")
    graph.add_edge("summarize_if_needed", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


# ─── Public Interface ─────────────────────────────────────────────────────────

async def create_agent():
    """
    Creates a persistent agent with SQLite checkpointing.
    This is the fix for Failure 3 (Context Collapse).
    """
    checkpointer = AsyncSqliteSaver.from_conn_string(settings.SQLITE_PATH)
    return build_graph(checkpointer=checkpointer)


async def run_agent(agent, user_message: str, thread_id: str = "default") -> str:
    """
    Run one turn of the agent.

    Args:
        agent:        Compiled LangGraph graph
        user_message: User's natural language query
        thread_id:    Conversation thread ID — all turns with same ID share state
                      (persisted via SQLite checkpointer)
    """
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: ProcurementState = {
        "messages": [HumanMessage(content=user_message)],
        "query_intent": None,
        "supplier_records": [],
        "contract_chunks": [],
        "risk_flags": [],
        "approval_chain": [],
        "approval_id": None,
        "approval_status": None,
        "turn_count": 0,
        "summary": None,
        "trace_id": None,
        "latency_ms": None,
        "tool_calls_made": [],
        "last_error": None,
        "retry_count": 0,
    }

    result = await agent.ainvoke(initial_state, config=config)
    last_ai_msg = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    return last_ai_msg.content if last_ai_msg else "No response generated."
