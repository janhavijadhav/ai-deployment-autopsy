"""
Shared pytest fixtures for the AI Deployment Autopsy test suite.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

# Point all DB/config at temp files during tests
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used-in-unit-tests")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "")


@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def tmp_sqlite(tmp_path):
    """Temporary SQLite database pre-populated with supplier schema."""
    db_path = str(tmp_path / "test_sap.db")
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE suppliers (
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
            CREATE TABLE contracts (
                contract_id        TEXT PRIMARY KEY,
                supplier_id        TEXT NOT NULL,
                contract_value_usd REAL NOT NULL,
                start_date         TEXT NOT NULL,
                expiry_date        TEXT NOT NULL,
                auto_renewal       INTEGER NOT NULL DEFAULT 0,
                notice_period_days INTEGER NOT NULL DEFAULT 30,
                governing_law      TEXT,
                signed_date        TEXT,
                contract_type      TEXT,
                pdf_filename       TEXT
            );
            INSERT INTO suppliers VALUES
                ('SUP-0001','Apex Industries','CN',0.82,3,78.5,'2024-01-10','electronics',1200000.0),
                ('SUP-0002','Nordic Materials','DE',0.21,5,96.2,'2024-03-15','raw_materials',800000.0),
                ('SUP-0003','Pacific Components','MX',0.55,2,88.0,'2023-11-20','electronics',450000.0);
            INSERT INTO contracts VALUES
                ('CTR-00001','SUP-0001',500000.0,'2023-01-01','2025-12-31',1,30,'NY','2022-12-15','MSA',NULL),
                ('CTR-00002','SUP-0002',200000.0,'2022-06-01','2024-05-31',0,60,'DE','2022-05-20','SOW',NULL);
        """)
        await db.commit()
    return db_path


@pytest.fixture
def sample_contract_text():
    """A realistic contract excerpt containing a penalty table."""
    return """
MASTER SUPPLY AGREEMENT — APEX INDUSTRIES LTD
Effective Date: January 1, 2023

SECTION 9.3 — LATE DELIVERY PENALTIES

If Supplier fails to deliver by the Delivery Date, Supplier shall pay liquidated damages:

| Delay Period      | Penalty Rate  | Maximum Cap     |
|-------------------|---------------|-----------------|
| Day 1 through 30  | 0.5% per day  | 10% of PO Value |
| Day 31 through 60 | 0.75% per day | 15% of PO Value |
| Day 61 and beyond | 1.0% per day  | 20% of PO Value |
| Force Majeure     | Waived        | N/A             |

SECTION 10.1 — PAYMENT TERMS

Payment terms are Net-30 from date of invoice. Late payment incurs 1.5% monthly interest.

SECTION 18 — GOVERNING LAW

This Agreement is governed by the laws of the State of New York.
"""
