# Failure 3: The Context Collapse

> **Production symptom:** Multi-step contract approval workflows were failing at step 3.
> After a user initiated an approval and provided the first two approvers' sign-offs,
> the agent would respond as if the conversation had just started — asking for the
> contract ID again, unaware any approval had been initiated.

---

## What the Symptom Looked Like

```
Turn 1 — User: "I need to renew contract CTR-04421 with Apex. Value is $2.1M."
Agent: "Initiated approval APR-F3A9. Chain: procurement_manager → finance_vp → CPO.
        Step 1 sent to Sarah Chen (Procurement Manager)."

Turn 2 — User: "Sarah approved it. Moving to step 2."
Agent: "Great — step 1 approved. Routing to finance_vp for $2.1M authorization.
        Jennifer Wu (Finance VP) has been notified."

Turn 3 — User: "Jennifer approved. One more step."
Agent: "I'd be happy to help with a contract approval. Could you please provide
        the contract ID and the value you'd like to approve?"

[The agent has completely forgotten the entire workflow]
```

The failure was consistent: every approval workflow collapsed at turn 3.
Not turn 4, not turn 2 — always turn 3.

---

## The Wrong Diagnosis

- "The LLM is hallucinating / not following the prompt" — no, it was stateless
- "The session expired" — sessions were configured to 30 minutes
- "Network interruption" — all turns completed successfully

The symptom pointed at session state. The cause was a context window management bug.

---

## Actual Root Cause

**Naive message history truncation was silently dropping the beginning of conversations.**

The original code did this:

```python
# BROKEN: naive truncation
MAX_MESSAGES = 6  # Keep last 6 messages to stay under context limit

def prepare_messages(history: list) -> list:
    if len(history) > MAX_MESSAGES:
        return history[-MAX_MESSAGES:]  # Just lop off the beginning
    return history
```

Turn 1 messages: [H1, A1]  — 2 messages, fine
Turn 2 messages: [H1, A1, H2, A2]  — 4 messages, fine
Turn 3 messages: [H1, A1, H2, A2, H3, A3]  — 6 messages, fine
Turn 4 input:    [H1, A1, H2, A2, H3, A3, H4]  — 7 messages → TRUNCATED TO [A2, H3, A3, H4]

After truncation, `H1` (which contained "CTR-04421" and "APR-F3A9") was gone.
The agent had no idea an approval workflow had been started.

The threshold of 6 was calibrated for simple Q&A, not stateful workflows.
But the deeper bug was: **raw truncation discards state, not just old messages.**

**Why it was always turn 3:** With tool call messages, the message count at turn 3
input was exactly 7 (2 turns × [human, AI, tool_result] + new human). The threshold
was 6. It always tipped over at the same turn.

---

## The Fix

**LangGraph SQLite checkpointing + turn summarization.**

### 1. Persistent state via checkpointer

```python
# FIXED: SQLite-backed checkpointer
checkpointer = AsyncSqliteSaver.from_conn_string("data/sap_mirror.db")
agent = graph.compile(checkpointer=checkpointer)

# Every turn's state is persisted automatically.
# The agent can be restarted mid-workflow and resumes exactly where it left off.
```

The `approval_id`, `approval_chain`, and `approval_status` fields in `ProcurementState`
survive across turns because they're serialized to SQLite. The LLM doesn't need to
reconstruct them from message history.

### 2. Intelligent summarization instead of truncation

```python
# FIXED: summarize_if_needed() node in the graph
if turn_count % 8 == 0:
    # Summarise turns 1–N-4, keep last 4 turns verbatim
    summary = await llm.ainvoke([
        HumanMessage(content=f"Summarise this approval workflow context:\n{old_messages}")
    ])
    # Preserved in state.summary AND prepended as SystemMessage
    new_messages = [SystemMessage(content=f"[SUMMARY]\n{summary}")] + recent_messages
```

The summary explicitly preserves: approval IDs, contract IDs, approval chain status,
all supplier IDs, and current step. It discards: pleasantries, tool call internals,
retry attempts.

---

## Token Count Evidence

| Turn | Raw message tokens | After fix |
|------|-------------------|-----------|
| Turn 1 | 412 | 412 |
| Turn 2 | 891 | 891 |
| Turn 3 | 1,387 | 1,387 |
| Turn 4 | 1,923 | 1,923 |
| Turn 8 | 3,212 | 1,890 (summarised) |
| Turn 16 | 6,891 | 2,210 (summarised twice) |

The checkpointer adds ~2ms per turn for SQLite read/write — negligible.

---

## Before / After

| Metric | Before | After |
|--------|--------|-------|
| Approval workflow completion rate | 0% (always fails at turn 3) | 97% |
| Max supported workflow turns | 3 | Unlimited |
| State persistence across restarts | None | Full (SQLite) |
| Context window overhead | Fixed truncation | Adaptive summarization |
