"""
Data layer: DuckDB (analytics) + SQLite (SAP mirror / checkpointer).

Two-database pattern mirrors real enterprise AI deployments:
- DuckDB: columnar OLAP queries (spend analytics, risk aggregations)
- SQLite: row-level OLTP + LangGraph state checkpointing

Schema uses EXACT column names that the schema drift monitor tracks.
Failure 4 demo: rename supplier_code → supplier_id to simulate a migration.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any

import aiosqlite
import duckdb

from src.config import settings


# ─── DuckDB (analytics queries) ──────────────────────────────────────────────

def get_duckdb() -> duckdb.DuckDBPyConnection:
    """Open DuckDB connection (thread-local)."""
    return duckdb.connect(settings.DUCKDB_PATH)


async def query_analytics(
    metric: str,
    period: str = "last_30_days",
    supplier_id: str | None = None,
) -> dict[str, Any]:
    """Run analytical queries against DuckDB procurement data warehouse."""

    period_filter = _period_to_sql_filter(period)
    supplier_filter = f"AND supplier_id = '{supplier_id}'" if supplier_id else ""

    queries = {
        "total_spend": f"""
            SELECT
                SUM(amount_usd) AS total_spend_usd,
                COUNT(*) AS transaction_count,
                AVG(amount_usd) AS avg_transaction_usd
            FROM purchase_orders
            WHERE {period_filter} {supplier_filter}
        """,
        "savings": f"""
            SELECT
                SUM(negotiated_savings_usd) AS realized_savings_usd,
                SUM(amount_usd) AS gross_spend_usd,
                ROUND(SUM(negotiated_savings_usd) / NULLIF(SUM(amount_usd), 0) * 100, 2) AS savings_pct
            FROM purchase_orders
            WHERE {period_filter} {supplier_filter}
        """,
        "risk_trend": f"""
            SELECT
                DATE_TRUNC('week', assessment_date) AS week,
                AVG(risk_score) AS avg_risk_score,
                COUNT(*) AS suppliers_assessed,
                SUM(CASE WHEN risk_score >= 0.7 THEN 1 ELSE 0 END) AS high_risk_count
            FROM supplier_risk_history
            WHERE {period_filter} {supplier_filter}
            GROUP BY 1
            ORDER BY 1
        """,
        "contract_expiry_forecast": f"""
            SELECT
                supplier_id,
                contract_id,
                contract_value_usd,
                expiry_date,
                DATEDIFF('day', CURRENT_DATE, expiry_date) AS days_until_expiry
            FROM contracts
            WHERE expiry_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL 90 DAYS
                  {supplier_filter}
            ORDER BY expiry_date
        """,
    }

    if metric not in queries:
        return {"error": f"Unknown metric '{metric}'. Valid: {list(queries.keys())}"}

    conn = get_duckdb()
    try:
        result = conn.execute(queries[metric]).fetchdf()
        return {"metric": metric, "period": period, "data": result.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e), "metric": metric}
    finally:
        conn.close()


# ─── SQLite (SAP mirror + supplier CRUD) ─────────────────────────────────────

@asynccontextmanager
async def get_sqlite():
    """Async SQLite connection context manager."""
    async with aiosqlite.connect(settings.SQLITE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def get_supplier(supplier_id: str) -> dict | None:
    """
    Fetch a supplier record by ID.

    IMPORTANT — Failure 4 note:
    This query uses the column name 'supplier_id'. If a schema migration renames
    this column to 'supplier_code', this query silently returns None for ALL suppliers.
    The schema_monitor.py catches this before it reaches production.
    """
    async with get_sqlite() as db:
        async with db.execute(
            """
            SELECT
                supplier_id,
                name,
                country,
                risk_score,
                active_contracts,
                on_time_delivery_pct,
                last_audit_date,
                procurement_category,
                annual_spend_usd
            FROM suppliers
            WHERE supplier_id = ?
            """,
            (supplier_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


async def search_suppliers(
    country: str | None = None,
    min_risk: float | None = None,
    max_risk: float | None = None,
    category: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """Search suppliers by criteria."""
    conditions = ["1=1"]
    params = []

    if country:
        conditions.append("country = ?")
        params.append(country)
    if min_risk is not None:
        conditions.append("risk_score >= ?")
        params.append(min_risk)
    if max_risk is not None:
        conditions.append("risk_score <= ?")
        params.append(max_risk)
    if category:
        conditions.append("procurement_category = ?")
        params.append(category)

    params.append(limit)
    where = " AND ".join(conditions)

    async with get_sqlite() as db:
        async with db.execute(
            f"SELECT * FROM suppliers WHERE {where} ORDER BY risk_score DESC LIMIT ?",
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]


async def get_contract_metadata(contract_id: str) -> dict | None:
    """Fetch contract metadata (not content — content is in Qdrant)."""
    async with get_sqlite() as db:
        async with db.execute(
            """
            SELECT
                contract_id,
                supplier_id,
                contract_value_usd,
                start_date,
                expiry_date,
                auto_renewal,
                notice_period_days,
                governing_law,
                signed_date,
                contract_type
            FROM contracts
            WHERE contract_id = ?
            """,
            (contract_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None


# ─── Schema initialisation ────────────────────────────────────────────────────

async def init_db() -> None:
    """Create database schema. Idempotent."""
    async with get_sqlite() as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS suppliers (
                supplier_id          TEXT PRIMARY KEY,
                name                 TEXT NOT NULL,
                country              TEXT NOT NULL,
                risk_score           REAL NOT NULL DEFAULT 0.0,
                active_contracts     INTEGER NOT NULL DEFAULT 0,
                on_time_delivery_pct REAL NOT NULL DEFAULT 100.0,
                last_audit_date      TEXT,
                procurement_category TEXT,
                annual_spend_usd     REAL
            );

            CREATE TABLE IF NOT EXISTS contracts (
                contract_id          TEXT PRIMARY KEY,
                supplier_id          TEXT NOT NULL REFERENCES suppliers(supplier_id),
                contract_value_usd   REAL NOT NULL,
                start_date           TEXT NOT NULL,
                expiry_date          TEXT NOT NULL,
                auto_renewal         INTEGER NOT NULL DEFAULT 0,
                notice_period_days   INTEGER NOT NULL DEFAULT 30,
                governing_law        TEXT,
                signed_date          TEXT,
                contract_type        TEXT,
                pdf_filename         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_contracts_supplier ON contracts(supplier_id);
            CREATE INDEX IF NOT EXISTS idx_suppliers_risk ON suppliers(risk_score);
            CREATE INDEX IF NOT EXISTS idx_suppliers_country ON suppliers(country);
        """)
        await db.commit()

    # DuckDB schema
    conn = get_duckdb()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS purchase_orders (
            po_id           VARCHAR PRIMARY KEY,
            supplier_id     VARCHAR NOT NULL,
            amount_usd      DOUBLE NOT NULL,
            order_date      DATE NOT NULL,
            category        VARCHAR,
            negotiated_savings_usd DOUBLE DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS supplier_risk_history (
            id              INTEGER,
            supplier_id     VARCHAR NOT NULL,
            risk_score      DOUBLE NOT NULL,
            assessment_date DATE NOT NULL
        )
    """)
    conn.close()


def _period_to_sql_filter(period: str) -> str:
    filters = {
        "last_7_days":    "order_date >= CURRENT_DATE - INTERVAL 7 DAYS",
        "last_30_days":   "order_date >= CURRENT_DATE - INTERVAL 30 DAYS",
        "last_quarter":   "order_date >= CURRENT_DATE - INTERVAL 90 DAYS",
        "ytd":            "YEAR(order_date) = YEAR(CURRENT_DATE)",
    }
    return filters.get(period, filters["last_30_days"])
