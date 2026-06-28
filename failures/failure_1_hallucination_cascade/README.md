# Failure 1: The Hallucination Cascade

> **Production symptom:** Procurement analysts using the agent to look up contract penalty
> clauses were receiving confident, specific answers with completely wrong dollar amounts.
> One analyst nearly approved a supplier waiver based on a fabricated "2% per week" penalty
> that was actually "0.5% per day" — a 10× difference.

---

## What the Symptom Looked Like

```
User: What are the late delivery penalties in the Apex Industries contract?

Agent: Based on the contract, Apex Industries is subject to a 2% per week penalty
       for late deliveries, capped at 25% of the PO value. The penalty clock starts
       after 48 hours of delay. Payment must be made within 15 business days.

[ACTUAL CONTRACT TEXT]: "0.5% of PO value per calendar day of delay, capped at 10% of
total PO value, applied after 24-hour grace period."
```

The agent answered with high confidence. No hedging. Every number was wrong.

**Measured faithfulness score: 34%**
(LLM-as-judge: only 34% of answers contained only claims supported by retrieved context)

---

## The Wrong Diagnosis (What a Junior Engineer Would Check)

- "Maybe the LLM is confabulating — try a different model or lower temperature"
- "Maybe the prompt doesn't tell it to only use retrieved context"
- "Maybe retrieval isn't finding the right contract"

All of these miss the actual problem. Changing the model helped slightly but didn't fix it.
Adding "only use context provided" to the prompt helped slightly but didn't fix it.
Retrieval WAS finding the right contract — that was the trap.

---

## Actual Root Cause

**Naive character-based chunking was splitting contract tables across chunk boundaries.**

The pricing penalty table in the PDF looked like this:

```
| Delay Period      | Penalty Rate | Cap        |
|-------------------|--------------|------------|
| Day 1–30          | 0.5%/day     | 10% of PO  |
| Day 31–60         | 0.75%/day    | 15% of PO  |
| Day 60+           | 1.0%/day     | 20% of PO  |
| Payment terms     | Net-30       | —          |
```

With 800-character chunk size, the chunker split this table at character 800,
right through the middle of the second row:

**Chunk A** (retrieved for penalty query):
```
| Delay Period      | Penalty Rate | Cap        |
|-------------------|--------------|------------|
| Day 1–30          | 0.5%/day     | 10% of PO  |
| Day 31–60         | 0.75%/
```

**Chunk B** (not retrieved — different embedding):
```
day     | 15% of PO  |
| Day 60+           | 1.0%/day     | 20% of PO  |
| Payment terms     | Net-30       | —          |
```

The LLM received Chunk A — a truncated table. Faced with an incomplete structure,
it hallucinated the completion based on training data patterns for penalty clauses.
That's where the "2% per week" came from: a plausible-sounding value from contract
training data, not from this contract.

**The retriever was doing its job correctly. The chunker was corrupting the data
before retrieval even ran.**

---

## The Fix

**Table-aware chunking**: detect tables in PDFs using PyMuPDF's block-level layout
analysis. Tables are atomic chunks — never split regardless of size.

See: [`src/rag/chunking.py`](../../src/rag/chunking.py) — `TableAwareChunker` class

Key changes:
1. `_classify_text_block()` detects pipe-separated or tab-separated content
2. Table blocks → single chunk, always
3. Text blocks → sentence-boundary splitting (not character count)
4. Table chunks get 1.25× score boost in retrieval for structured queries

```python
# BROKEN: naive character split
chunk = text[start:start + 800]   # Table split here at char 800

# FIXED: table-aware chunking
if chunk_type == ChunkType.TABLE:
    # Tables are atomic — never split
    chunks.append(DocumentChunk(content=block_text, chunk_type=ChunkType.TABLE, ...))
```

---

## Before / After Metrics

| Metric | Before (naive chunking) | After (table-aware) |
|--------|------------------------|---------------------|
| Faithfulness score | 34% | 91% |
| Table recall | 12% | 94% |
| Hallucinated prices | 8/10 queries | 0/10 queries |
| Chunking time | 120ms | 180ms (+50ms) |
| Avg chunk quality | 0.38 | 0.81 |

The 50ms chunking overhead happens at ingest time (once), not query time.

---

## LLM-as-Judge Eval (Faithfulness)

```python
# llm_judge.py — faithfulness scoring
score = await judge.score_faithfulness(
    question="What are the late delivery penalties?",
    answer=agent_response,
    context=retrieved_chunks,
)
# Before fix: score = 0.34
# After fix:  score = 0.91
```

The faithfulness judge now runs in CI on every PR (`make eval-faithfulness`).
A PR that drops faithfulness below 0.85 is blocked.

---

## Broken Code (kept for reference)

See: [`broken_chunking.py`](./broken_chunking.py)
