"""
Triggers the schema drift bomb — renames suppliers.supplier_id → suppliers.supplier_code.

This simulates the real production migration that caused Failure 4.
Run AFTER make schema-snapshot to see the drift detector catch it.

Usage:
  python failures/failure_4_schema_drift_bomb/trigger_migration.py
  make schema-diff  # Should detect drift and exit 1
"""

import asyncio
import sys

sys.path.insert(0, ".")

import aiosqlite
from src.config import settings


async def apply_migration():
    """
    Simulates a database team's schema normalization migration.
    Renames supplier_id → supplier_code.

    In the real incident, this was part of a larger migration script
    titled "normalize_pk_naming_convention.sql" run by the DB team.
    The AI team was not on the change notification list.
    """
    print("Applying migration: suppliers.supplier_id → suppliers.supplier_code")

    async with aiosqlite.connect(settings.SQLITE_PATH) as db:
        # SQLite doesn't support ALTER TABLE RENAME COLUMN in older versions,
        # so we do it the proper way: create new table, copy, drop old.
        await db.executescript("""
            -- Step 1: Create new table with renamed column
            CREATE TABLE IF NOT EXISTS suppliers_new (
                supplier_code        TEXT PRIMARY KEY,   -- RENAMED from supplier_id
                name                 TEXT NOT NULL,
                country              TEXT NOT NULL,
                risk_score           REAL NOT NULL DEFAULT 0.0,
                active_contracts     INTEGER NOT NULL DEFAULT 0,
                on_time_delivery_pct REAL NOT NULL DEFAULT 100.0,
                last_audit_date      TEXT,
                procurement_category TEXT,
                annual_spend_usd     REAL
            );

            -- Step 2: Copy data with renamed column
            INSERT INTO suppliers_new
            SELECT
                supplier_id AS supplier_code,
                name, country, risk_score, active_contracts,
                on_time_delivery_pct, last_audit_date,
                procurement_category, annual_spend_usd
            FROM suppliers;

            -- Step 3: Drop old table
            DROP TABLE suppliers;

            -- Step 4: Rename new table
            ALTER TABLE suppliers_new RENAME TO suppliers;
        """)
        await db.commit()

    print("Migration complete.")
    print()
    print("The agent will now return 'Supplier not found' for ALL supplier queries.")
    print("This is because src/data/database.py still queries: supplier_id = ?")
    print()
    print("Run: make schema-diff")
    print("Expected output: DRIFT DETECTED — COLUMN_DROPPED: suppliers.supplier_id")
    print()
    print("To revert: make seed  (re-seeds with original schema)")


async def main():
    await apply_migration()


if __name__ == "__main__":
    asyncio.run(main())
