"""
FastAPI entrypoint for the Procurement Intelligence Agent.

Endpoints:
  POST /query          — run one agent turn (thread_id in body for multi-turn)
  GET  /health         — liveness check
  GET  /ready          — readiness (checks Qdrant + Redis + DB)
  GET  /metrics        — Prometheus metrics (also scraped on port 9091)
  POST /ingest         — trigger contract ingest (admin)
  POST /schema/snapshot — take schema snapshot
  GET  /schema/diff    — detect schema drift vs last snapshot
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import make_asgi_app

from src.config import settings
from src.agent.procurement_agent import create_agent, run_agent
from src.data.database import init_db
from src.data.schema_monitor import take_snapshot, detect_drift
from src.observability.tracing import metrics, tracer
from src.rag.pipeline import ingest_contracts


# ─── Lifespan ─────────────────────────────────────────────────────────────────

_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, warm agent, start Prometheus scrape server."""
    global _agent

    print("[startup] Initialising database schema...")
    await init_db()

    print("[startup] Warming LangGraph agent (SQLite checkpointer)...")
    _agent = await create_agent()

    print("[startup] Starting Prometheus metrics server...")
    metrics.start_metrics_server()

    print(f"[startup] Ready. Agent: {settings.CLAUDE_MODEL}")
    yield

    print("[shutdown] Graceful shutdown complete.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Procurement Intelligence Agent",
    description="AI Deployment Autopsy — Meridian Manufacturing Corp",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    message: str
    thread_id: str = ""          # empty → new conversation
    operator_role: str = "Procurement Analyst"

class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    latency_ms: float
    trace_id: str

class IngestRequest(BaseModel):
    contracts_dir: str = "data/contracts/"

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str = "1.0.0"

class ReadyResponse(BaseModel):
    status: str
    checks: dict[str, str]


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Run one agent turn. Pass thread_id to continue an existing conversation
    (e.g. multi-step approval workflows). Omit for a new thread.
    """
    if _agent is None:
        raise HTTPException(503, "Agent not initialised — startup may still be running")

    thread_id = req.thread_id or f"thread-{uuid.uuid4().hex[:8]}"
    trace_id = str(uuid.uuid4())

    t0 = time.perf_counter()
    with tracer.span("api.query", attributes={"thread_id": thread_id, "trace_id": trace_id}):
        try:
            answer = await run_agent(_agent, user_message=req.message, thread_id=thread_id)
        except Exception as e:
            metrics.record_request(intent="unknown", status="error")
            raise HTTPException(500, f"Agent error: {e}") from e

    latency_ms = (time.perf_counter() - t0) * 1000
    metrics.record_request(intent="query", status="success")

    return QueryResponse(
        answer=answer,
        thread_id=thread_id,
        latency_ms=round(latency_ms, 1),
        trace_id=trace_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness probe — returns 200 as long as the process is running."""
    return HealthResponse(status="ok", model=settings.CLAUDE_MODEL)


@app.get("/ready", response_model=ReadyResponse)
async def ready():
    """
    Readiness probe — checks connectivity to Qdrant, Redis, and SQLite.
    Returns 503 if any dependency is unavailable.
    """
    checks: dict[str, str] = {}
    all_ok = True

    # Qdrant
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=settings.QDRANT_URL)
        await client.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"
        all_ok = False

    # Redis
    try:
        import redis.asyncio as aioredis
        r = aioredis.from_url(settings.REDIS_URL)
        await r.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"
        all_ok = False

    # SQLite
    try:
        import aiosqlite
        async with aiosqlite.connect(settings.SQLITE_PATH) as db:
            await db.execute("SELECT 1")
        checks["sqlite"] = "ok"
    except Exception as e:
        checks["sqlite"] = f"error: {e}"
        all_ok = False

    # Agent
    checks["agent"] = "ok" if _agent is not None else "not_ready"
    if _agent is None:
        all_ok = False

    status_code = 200 if all_ok else 503
    response = ReadyResponse(
        status="ready" if all_ok else "degraded",
        checks=checks,
    )
    if not all_ok:
        raise HTTPException(status_code=status_code, detail=response.model_dump())
    return response


@app.post("/ingest")
async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger contract ingest into Qdrant (runs in background).
    Called after adding new PDF contracts to data/contracts/.
    Also invalidates the semantic cache for the contracts namespace.
    """
    async def _run_ingest():
        from src.cache.redis_cache import SemanticCache
        stats = await ingest_contracts(req.contracts_dir)
        cache = SemanticCache()
        await cache.invalidate("contracts")
        print(f"[ingest] Complete: {stats}")

    background_tasks.add_task(_run_ingest)
    return {"status": "ingest_started", "contracts_dir": req.contracts_dir}


@app.post("/schema/snapshot")
async def schema_snapshot(label: str = "manual"):
    """Take a schema snapshot. Run before any database migration."""
    path = await take_snapshot(label=label)
    return {"status": "ok", "snapshot_path": str(path)}


@app.get("/schema/diff")
async def schema_diff():
    """
    Detect schema drift vs most recent snapshot.
    Returns 409 if drift is detected (mirrors CI gate exit code 1 behaviour).
    """
    result = await detect_drift()
    if result["drifted"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "Schema drift detected",
                "changes": result["changes"],
                "baseline_captured_at": result.get("baseline_captured_at"),
            },
        )
    return {"status": "no_drift", "baseline_label": result.get("baseline_label")}
