"""
BROKEN: Sequential tool calls — causes Failure 2 (Latency Wall).

This demonstrates the original sequential execution pattern.
Wall time: 12–16 seconds.

See src/agent/procurement_agent.py for the fixed parallel async pattern.
"""

import asyncio
import time


# Simulate tool latencies (actual measured in production)
async def _fake_search_contracts(query: str) -> list[dict]:
    await asyncio.sleep(2.58)  # Qdrant search + embedding computation
    return [{"chunk_id": "CTR-001-001", "content": "...penalty clause...", "score": 0.91}]


async def _fake_lookup_supplier(supplier_id: str) -> dict:
    await asyncio.sleep(2.29)  # SQLite query + network
    return {"supplier_id": supplier_id, "name": "Apex Industries", "risk_score": 0.42}


async def _fake_flag_risks(supplier_ids: list[str]) -> list[dict]:
    await asyncio.sleep(2.84)  # Multiple DB queries
    return [{"flag_id": "RISK-001", "severity": "medium", "category": "delivery_delay"}]


# ─── BROKEN: Sequential execution ────────────────────────────────────────────

async def broken_sequential_query(query: str, supplier_id: str):
    """
    BROKEN pattern: tools run one-after-another.
    Each tool waits for the previous to finish — even though they're INDEPENDENT.

    Wall time: sum of all tool latencies = 2.58 + 2.29 + 2.84 = 7.71s
    Plus LLM synthesis: ~4.5s
    Total: ~12-14 seconds
    """
    t0 = time.perf_counter()
    print(f"\n[BROKEN] Sequential execution")

    print(f"  {time.perf_counter()-t0:.2f}s — search_contracts started")
    contracts = await _fake_search_contracts(query)
    print(f"  {time.perf_counter()-t0:.2f}s — search_contracts done")

    # BUG: This tool doesn't NEED contracts to be done first
    # It's just waiting because it's next in the queue
    print(f"  {time.perf_counter()-t0:.2f}s — lookup_supplier started")
    supplier = await _fake_lookup_supplier(supplier_id)
    print(f"  {time.perf_counter()-t0:.2f}s — lookup_supplier done")

    # BUG: Same — completely independent but waiting
    print(f"  {time.perf_counter()-t0:.2f}s — flag_risks started")
    risks = await _fake_flag_risks([supplier_id])
    print(f"  {time.perf_counter()-t0:.2f}s — flag_risks done")

    total = time.perf_counter() - t0
    print(f"\n  Tool wall time: {total:.2f}s")
    return contracts, supplier, risks


# ─── FIXED: Parallel execution ────────────────────────────────────────────────

async def fixed_parallel_query(query: str, supplier_id: str):
    """
    FIXED pattern: all independent tools run in parallel with asyncio.gather.

    Wall time: max of all tool latencies = max(2.58, 2.29, 2.84) = 2.84s
    Plus LLM synthesis: ~0.8s (cached context reduces synthesis time)
    Total: ~3.6 seconds
    """
    t0 = time.perf_counter()
    print(f"\n[FIXED] Parallel execution")
    print(f"  {time.perf_counter()-t0:.2f}s — all 3 tools started in parallel")

    contracts, supplier, risks = await asyncio.gather(
        _fake_search_contracts(query),
        _fake_lookup_supplier(supplier_id),
        _fake_flag_risks([supplier_id]),
    )

    total = time.perf_counter() - t0
    print(f"  {total:.2f}s — all 3 tools done")
    print(f"\n  Tool wall time: {total:.2f}s")
    return contracts, supplier, risks


async def main():
    print("Latency Wall — Failure 2 Demonstration")
    print("=" * 50)

    await broken_sequential_query("penalty clauses", "SUP-0001")
    await fixed_parallel_query("penalty clauses", "SUP-0001")

    print("\nSummary:")
    print("  Sequential: ~7.7s tool time + ~4.5s LLM = ~12.2s total")
    print("  Parallel:   ~2.8s tool time + ~0.8s LLM = ~3.6s total")
    print("  Improvement: 3.4×")


if __name__ == "__main__":
    asyncio.run(main())
