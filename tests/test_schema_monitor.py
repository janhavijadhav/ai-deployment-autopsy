"""
Tests for Failure 4 fix: schema drift detection.

Key assertions:
- Snapshot captures current schema correctly
- Dropping a column is detected as CRITICAL drift
- Adding a column is detected as INFO
- Renaming a column (drop + add) is detected
- No drift when schema is unchanged
- detect_drift() exits clean on matching schemas
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

from unittest.mock import patch


class TestSchemaDriftDetection:

    @pytest_asyncio.fixture
    async def db_with_schema(self, tmp_path):
        """Fresh SQLite DB with known schema."""
        db_path = str(tmp_path / "monitor_test.db")
        async with aiosqlite.connect(db_path) as db:
            await db.executescript("""
                CREATE TABLE suppliers (
                    supplier_id   TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    country       TEXT NOT NULL,
                    risk_score    REAL DEFAULT 0.0
                );
            """)
            await db.commit()
        return db_path

    @pytest_asyncio.fixture
    async def snapshot_dir(self, tmp_path):
        return str(tmp_path / "snapshots")

    def _make_schema(self, columns: dict) -> dict:
        return {"suppliers": columns}

    def test_no_drift_on_identical_schemas(self):
        """Identical baseline and current → empty changes list."""
        from src.data.schema_monitor import _diff_schemas
        schema = {
            "suppliers": {
                "supplier_id": {"type": "TEXT", "notnull": True, "default": None, "primary_key": True},
                "name": {"type": "TEXT", "notnull": True, "default": None, "primary_key": False},
            }
        }
        changes = _diff_schemas(schema, schema)
        assert changes == []

    def test_column_drop_detected_as_critical(self):
        """Removing supplier_id is CRITICAL drift."""
        from src.data.schema_monitor import _diff_schemas
        baseline = {
            "suppliers": {
                "supplier_id": {"type": "TEXT", "notnull": True, "default": None, "primary_key": True},
                "name": {"type": "TEXT", "notnull": True, "default": None, "primary_key": False},
            }
        }
        current = {
            "suppliers": {
                "name": {"type": "TEXT", "notnull": True, "default": None, "primary_key": False},
            }
        }
        changes = _diff_schemas(baseline, current)
        assert len(changes) == 1
        assert changes[0]["change"] == "COLUMN_DROPPED"
        assert changes[0]["column"] == "supplier_id"
        assert changes[0]["severity"] == "CRITICAL"

    def test_column_rename_detected(self):
        """
        The exact Failure 4 scenario: supplier_id renamed to supplier_code.
        Appears as COLUMN_DROPPED (critical) + COLUMN_ADDED (info).
        """
        from src.data.schema_monitor import _diff_schemas
        baseline = {
            "suppliers": {
                "supplier_id": {"type": "TEXT", "notnull": True, "default": None, "primary_key": True},
            }
        }
        current = {
            "suppliers": {
                "supplier_code": {"type": "TEXT", "notnull": True, "default": None, "primary_key": True},
            }
        }
        changes = _diff_schemas(baseline, current)
        change_types = {c["change"] for c in changes}
        assert "COLUMN_DROPPED" in change_types
        assert "COLUMN_ADDED" in change_types

        dropped = next(c for c in changes if c["change"] == "COLUMN_DROPPED")
        assert dropped["column"] == "supplier_id"
        assert dropped["severity"] == "CRITICAL"
        assert "impact" in dropped  # Must explain the blast radius

    def test_column_add_detected_as_info(self):
        """Adding a new column is INFO — not breaking."""
        from src.data.schema_monitor import _diff_schemas
        baseline = {"suppliers": {"supplier_id": {"type": "TEXT", "notnull": True, "default": None, "primary_key": True}}}
        current = {
            "suppliers": {
                "supplier_id": {"type": "TEXT", "notnull": True, "default": None, "primary_key": True},
                "new_column": {"type": "TEXT", "notnull": False, "default": None, "primary_key": False},
            }
        }
        changes = _diff_schemas(baseline, current)
        assert len(changes) == 1
        assert changes[0]["change"] == "COLUMN_ADDED"
        assert changes[0]["severity"] == "INFO"

    def test_type_change_detected_as_high(self):
        """Changing a column type is HIGH severity."""
        from src.data.schema_monitor import _diff_schemas
        baseline = {"suppliers": {"risk_score": {"type": "REAL", "notnull": False, "default": "0.0", "primary_key": False}}}
        current  = {"suppliers": {"risk_score": {"type": "TEXT", "notnull": False, "default": None, "primary_key": False}}}
        changes = _diff_schemas(baseline, current)
        assert len(changes) == 1
        assert changes[0]["change"] == "TYPE_CHANGED"
        assert changes[0]["severity"] == "HIGH"
        assert changes[0]["was_type"] == "REAL"
        assert changes[0]["now_type"] == "TEXT"

    def test_table_drop_detected_as_critical(self):
        """Dropping an entire table is CRITICAL."""
        from src.data.schema_monitor import _diff_schemas
        baseline = {"suppliers": {}, "contracts": {}}
        current  = {"suppliers": {}}
        changes = _diff_schemas(baseline, current)
        assert any(c["change"] == "TABLE_DROPPED" and c["table"] == "contracts" for c in changes)

    def test_checksum_changes_on_drift(self):
        """Checksum of schema must differ when columns change."""
        from src.data.schema_monitor import _checksum
        schema_a = {"suppliers": {"supplier_id": {"type": "TEXT"}}}
        schema_b = {"suppliers": {"supplier_code": {"type": "TEXT"}}}
        assert _checksum(schema_a) != _checksum(schema_b)

    def test_checksum_stable_on_same_schema(self):
        """Same schema always produces the same checksum."""
        from src.data.schema_monitor import _checksum
        schema = {"suppliers": {"supplier_id": {"type": "TEXT", "notnull": True}}}
        assert _checksum(schema) == _checksum(schema)

    @pytest.mark.asyncio
    async def test_extract_schema_reads_sqlite(self, tmp_sqlite):
        """_extract_schema returns correct column names from a real SQLite DB."""
        from src.data.schema_monitor import _extract_schema
        with patch("src.data.schema_monitor.settings") as mock_settings:
            mock_settings.SQLITE_PATH = tmp_sqlite
            schema = await _extract_schema()

        assert "suppliers" in schema
        assert "supplier_id" in schema["suppliers"]
        assert schema["suppliers"]["supplier_id"]["primary_key"] is True
        assert "contracts" in schema
