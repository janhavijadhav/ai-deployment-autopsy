"""
Mock Supplier REST API
======================
Simulates the external supplier status API used in production.
Serves two endpoints consumed by SupplierAPIClient:

  POST /oauth/token          — OAuth2 client_credentials token endpoint
  GET  /suppliers/{id}/status — Current supplier status payload

This server runs in Docker alongside the main stack (see docker-compose.yml).
It deliberately simulates:
  - Token expiry (configurable TTL)
  - Occasional 500s (FAILURE_RATE env var, default 0.0)
  - Realistic latency (LATENCY_MS env var, default 50)

Failure 5 demonstration:
  Set ROTATE_ON_USE=true to make every GET /status call invalidate the token
  so the broken concurrent refresh code (broken_oauth.py) deadlocks.
"""

from __future__ import annotations

import asyncio
import os
import time
import random
import secrets
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "3600"))
FAILURE_RATE = float(os.getenv("FAILURE_RATE", "0.0"))       # 0.0–1.0
LATENCY_MS = int(os.getenv("LATENCY_MS", "50"))
ROTATE_ON_USE = os.getenv("ROTATE_ON_USE", "false").lower() == "true"
CLIENT_ID = os.getenv("SUPPLIER_CLIENT_ID", "meridian-client")
CLIENT_SECRET = os.getenv("SUPPLIER_CLIENT_SECRET", "s3cr3t")

# ---------------------------------------------------------------------------
# Token store (in-memory — single instance only)
# ---------------------------------------------------------------------------
_active_tokens: dict[str, float] = {}   # token → expires_at


def _issue_token() -> dict:
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + TOKEN_TTL_SECONDS
    _active_tokens[token] = expires_at
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": TOKEN_TTL_SECONDS,
    }


def _validate_token(authorization: Optional[str] = Header(None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    expires_at = _active_tokens.get(token)
    if expires_at is None or time.time() > expires_at:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
    return token


# ---------------------------------------------------------------------------
# Supplier data (static fixture — matches data/seed_data.py)
# ---------------------------------------------------------------------------
_SUPPLIERS: dict[str, dict] = {
    "SUP-0001": {
        "supplier_id": "SUP-0001",
        "name": "Apex Industries",
        "status": "active",
        "on_time_delivery_rate": 0.87,
        "quality_rejection_rate": 0.04,
        "financial_health_score": 0.72,
        "active_purchase_orders": 12,
        "open_disputes": 1,
        "last_audit_date": "2024-11-15",
        "certifications": ["ISO-9001", "ISO-14001"],
        "capacity_utilization_pct": 83,
    },
    "SUP-0002": {
        "supplier_id": "SUP-0002",
        "name": "Brightfield Components",
        "status": "active",
        "on_time_delivery_rate": 0.97,
        "quality_rejection_rate": 0.01,
        "financial_health_score": 0.91,
        "active_purchase_orders": 31,
        "open_disputes": 0,
        "last_audit_date": "2025-01-08",
        "certifications": ["ISO-9001", "IATF-16949", "AS9100"],
        "capacity_utilization_pct": 61,
    },
    "SUP-0003": {
        "supplier_id": "SUP-0003",
        "name": "Dalton Materials",
        "status": "probation",
        "on_time_delivery_rate": 0.71,
        "quality_rejection_rate": 0.09,
        "financial_health_score": 0.44,
        "active_purchase_orders": 3,
        "open_disputes": 4,
        "last_audit_date": "2024-08-22",
        "certifications": ["ISO-9001"],
        "capacity_utilization_pct": 45,
    },
}

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Meridian Supplier Mock API",
    description="Local mock for the supplier partner API used in integration tests.",
    version="1.0.0",
)


async def _inject_latency_and_faults():
    """Simulate network latency and random failures."""
    if LATENCY_MS > 0:
        await asyncio.sleep(LATENCY_MS / 1000)
    if random.random() < FAILURE_RATE:
        raise HTTPException(status_code=500, detail="Simulated upstream failure")


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------
class TokenRequest(BaseModel):
    grant_type: str
    client_id: str
    client_secret: str
    scope: Optional[str] = None


@app.post("/oauth/token")
async def issue_token(body: TokenRequest):
    await _inject_latency_and_faults()
    if body.grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="Unsupported grant_type")
    if body.client_id != CLIENT_ID or body.client_secret != CLIENT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid client credentials")
    return _issue_token()


# ---------------------------------------------------------------------------
# Supplier status endpoint
# ---------------------------------------------------------------------------
@app.get("/suppliers/{supplier_id}/status")
async def get_supplier_status(
    supplier_id: str,
    token: str = Depends(_validate_token),
):
    await _inject_latency_and_faults()

    supplier = _SUPPLIERS.get(supplier_id)
    if supplier is None:
        raise HTTPException(status_code=404, detail=f"Supplier {supplier_id!r} not found")

    if ROTATE_ON_USE:
        # Failure 5 demo: invalidate token after each use to force constant refresh
        _active_tokens.pop(token, None)

    return {
        **supplier,
        "retrieved_at": time.time(),
        "request_id": str(uuid.uuid4()),
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "token_ttl_seconds": TOKEN_TTL_SECONDS, "rotate_on_use": ROTATE_ON_USE}


@app.get("/")
async def root():
    return {"service": "Meridian Supplier Mock API", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
