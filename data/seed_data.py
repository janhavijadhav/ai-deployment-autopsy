"""
Seed fake SAP data: 50 suppliers, 120 contracts, 2000 purchase orders.
Run once: python -m data.seed_data   (or: make seed)
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import date, timedelta

import aiosqlite
import duckdb

# Ensure src is importable from project root
import sys
sys.path.insert(0, ".")

from src.config import settings
from src.data.database import init_db

random.seed(42)  # Reproducible fake data

COUNTRIES = ["CN", "DE", "MX", "IN", "US", "JP", "KR", "BR", "PL", "VN"]
CATEGORIES = ["electronics", "raw_materials", "packaging", "logistics", "chemicals", "tooling"]
NAMES_PREFIX = ["Apex", "Meridian", "Global", "Pacific", "Euro", "Sino", "Nordic", "Allied"]
NAMES_SUFFIX = ["Industries", "Manufacturing", "Supply Co", "Components", "Materials", "Tech"]


def rand_supplier_id(i: int) -> str:
    return f"SUP-{i:04d}"


def rand_name() -> str:
    return f"{random.choice(NAMES_PREFIX)} {random.choice(NAMES_SUFFIX)}"


def rand_date(start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()


async def seed_suppliers(db: aiosqlite.Connection) -> list[str]:
    """Insert 50 fake suppliers."""
    supplier_ids = []
    for i in range(1, 51):
        sid = rand_supplier_id(i)
        # ~15% are high risk (risk_score >= 0.7) to make risk queries interesting
        risk = round(random.gauss(0.35, 0.2), 3)
        risk = max(0.05, min(0.99, risk))
        if i % 7 == 0:  # Deliberately make some high-risk
            risk = round(random.uniform(0.7, 0.95), 3)

        await db.execute(
            """
            INSERT OR IGNORE INTO suppliers
            (supplier_id, name, country, risk_score, active_contracts,
             on_time_delivery_pct, last_audit_date, procurement_category, annual_spend_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sid,
                rand_name(),
                random.choice(COUNTRIES),
                risk,
                random.randint(1, 12),
                round(random.gauss(91, 8), 1),   # most ≥ 85%
                rand_date(date(2023, 1, 1), date(2024, 6, 1)),
                random.choice(CATEGORIES),
                round(random.uniform(500_000, 15_000_000), 2),
            ),
        )
        supplier_ids.append(sid)
    await db.commit()
    print(f"  ✓ Seeded {len(supplier_ids)} suppliers")
    return supplier_ids


async def seed_contracts(db: aiosqlite.Connection, supplier_ids: list[str]) -> list[str]:
    """Insert 120 fake contracts (2–3 per supplier on average)."""
    contract_ids = []
    contract_types = ["MSA", "SOW", "NDA", "PO_Framework", "SLA"]

    for i in range(120):
        cid = f"CTR-{i:05d}"
        sid = random.choice(supplier_ids)
        start = date(2022, 1, 1) + timedelta(days=random.randint(0, 730))
        duration_days = random.choice([365, 730, 1095])
        expiry = start + timedelta(days=duration_days)

        await db.execute(
            """
            INSERT OR IGNORE INTO contracts
            (contract_id, supplier_id, contract_value_usd, start_date, expiry_date,
             auto_renewal, notice_period_days, governing_law, signed_date, contract_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid, sid,
                round(random.uniform(50_000, 5_000_000), 2),
                start.isoformat(),
                expiry.isoformat(),
                random.choice([0, 0, 1]),   # 33% auto-renew
                random.choice([30, 60, 90]),
                random.choice(["NY", "DE", "CA", "England"]),
                (start - timedelta(days=random.randint(5, 30))).isoformat(),
                random.choice(contract_types),
            ),
        )
        contract_ids.append(cid)

    await db.commit()
    print(f"  ✓ Seeded {len(contract_ids)} contracts")
    return contract_ids


def seed_duckdb(supplier_ids: list[str]) -> None:
    """Insert 2000 purchase orders + 6 months of risk history into DuckDB."""
    conn = duckdb.connect(settings.DUCKDB_PATH)

    # Purchase orders
    po_rows = []
    for _ in range(2000):
        sid = random.choice(supplier_ids)
        order_date = date.today() - timedelta(days=random.randint(0, 180))
        amount = round(random.uniform(5_000, 500_000), 2)
        savings = round(amount * random.uniform(0.0, 0.12), 2)  # 0–12% savings
        po_rows.append((
            f"PO-{uuid.uuid4().hex[:8].upper()}",
            sid, amount, order_date.isoformat(),
            random.choice(CATEGORIES), savings,
        ))

    conn.execute("DELETE FROM purchase_orders")
    conn.executemany(
        "INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?)", po_rows
    )

    # Risk history (weekly snapshots for last 26 weeks)
    risk_rows = []
    for sid in supplier_ids:
        base_risk = random.uniform(0.1, 0.8)
        for week in range(26):
            assessment_date = date.today() - timedelta(weeks=week)
            # Small random walk
            risk = max(0.05, min(0.99, base_risk + random.gauss(0, 0.03)))
            risk_rows.append((random.randint(1, 999999), sid, round(risk, 3), assessment_date.isoformat()))

    conn.execute("DELETE FROM supplier_risk_history")
    conn.executemany(
        "INSERT INTO supplier_risk_history VALUES (?, ?, ?, ?)", risk_rows
    )

    conn.commit()
    conn.close()
    print(f"  ✓ Seeded 2000 POs + {len(risk_rows)} risk history rows in DuckDB")


async def main():
    print("Seeding fake SAP data...")
    await init_db()
    async with aiosqlite.connect(settings.SQLITE_PATH) as db:
        supplier_ids = await seed_suppliers(db)
        await seed_contracts(db, supplier_ids)
    seed_duckdb(supplier_ids)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
