"""
Schema Drift Monitor — the fix for Failure 4 (Schema Drift Bomb).

THE PROBLEM:
  An upstream Postgres schema migration renamed suppliers.supplier_code → suppliers.supplier_id.
  Nobody told the AI team. The agent worked perfectly for 3 weeks, then started returning
  None for every supplier lookup — silently. No error was raised. Users saw "Supplier not found"
  for every query, which was dismissed as a data issue for two days before anyone checked the schema.

THE FIX:
  1. Take a JSON snapshot of the full schema before any migration.
  2. After migration, run schema-diff to detect column additions, removals, renames, type changes.
  3. CI gate: if schema_monitor diff exits non-zero, the deploy is blocked.
  4. Alert webhook fires immediately on drift detection (before agent goes to prod).

Usage:
  python -m src.data.schema_monitor snapshot   # Before migration
  python -m src.data.schema_monitor diff       # After migration
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite
import httpx

from src.config import settings


# ─── Schema Snapshot ──────────────────────────────────────────────────────────

async def take_snapshot(label: str = "pre_migration") -> Path:
    """
    Capture current SQLite schema as a versioned JSON snapshot.
    Run this BEFORE any database migration.
    """
    snapshot_dir = Path(settings.SCHEMA_SNAPSHOT_PATH)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    schema = await _extract_schema()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = snapshot_dir / f"schema_{timestamp}_{label}.json"

    payload = {
        "captured_at": timestamp,
        "label": label,
        "db_path": settings.SQLITE_PATH,
        "schema": schema,
        "checksum": _checksum(schema),
    }

    snapshot_path.write_text(json.dumps(payload, indent=2))
    print(f"[schema_monitor] Snapshot saved: {snapshot_path}")
    return snapshot_path


async def detect_drift(baseline_path: str | None = None) -> dict[str, Any]:
    """
    Compare current schema against the most recent snapshot.
    Returns a dict with `drifted: bool` and a list of `changes`.

    Called by CI pipeline — exit code 1 if drift detected.
    """
    snapshot_dir = Path(settings.SCHEMA_SNAPSHOT_PATH)

    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text())
    else:
        # Use most recent snapshot
        snapshots = sorted(snapshot_dir.glob("schema_*.json"))
        if not snapshots:
            return {"drifted": False, "changes": [], "error": "No baseline snapshot found. Run: make schema-snapshot"}
        baseline = json.loads(snapshots[-1].read_text())

    current_schema = await _extract_schema()
    baseline_schema = baseline["schema"]

    changes = _diff_schemas(baseline_schema, current_schema)
    drifted = len(changes) > 0

    result = {
        "drifted": drifted,
        "baseline_label": baseline.get("label"),
        "baseline_captured_at": baseline.get("captured_at"),
        "current_checksum": _checksum(current_schema),
        "baseline_checksum": baseline.get("checksum"),
        "changes": changes,
    }

    if drifted:
        _print_drift_report(changes)
        await _fire_alert(result)

    return result


# ─── Schema extraction ────────────────────────────────────────────────────────

async def _extract_schema() -> dict[str, Any]:
    """Extract table + column definitions from SQLite."""
    schema: dict[str, Any] = {}
    async with aiosqlite.connect(settings.SQLITE_PATH) as db:
        async with db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ) as cur:
            tables = [row[0] async for row in cur]

        for table in tables:
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                cols = {}
                async for row in cur:
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    cols[row[1]] = {
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default": row[4],
                        "primary_key": bool(row[5]),
                    }
            schema[table] = cols

    return schema


def _diff_schemas(
    baseline: dict[str, Any],
    current: dict[str, Any],
) -> list[dict[str, str]]:
    """
    Produce a list of schema changes between baseline and current.
    Detects: table drops, table adds, column drops, column adds, type changes.
    """
    changes = []

    # Tables removed
    for table in baseline:
        if table not in current:
            changes.append({"change": "TABLE_DROPPED", "table": table, "severity": "CRITICAL"})

    # Tables added
    for table in current:
        if table not in baseline:
            changes.append({"change": "TABLE_ADDED", "table": table, "severity": "INFO"})

    # Column-level drift
    for table in baseline:
        if table not in current:
            continue
        base_cols = baseline[table]
        curr_cols = current[table]

        for col in base_cols:
            if col not in curr_cols:
                changes.append({
                    "change": "COLUMN_DROPPED",
                    "table": table,
                    "column": col,
                    "was_type": base_cols[col]["type"],
                    "severity": "CRITICAL",
                    "impact": (
                        f"Any query referencing {table}.{col} will now fail or return wrong results. "
                        "Check agent tools that query this column."
                    ),
                })

        for col in curr_cols:
            if col not in base_cols:
                changes.append({
                    "change": "COLUMN_ADDED",
                    "table": table,
                    "column": col,
                    "new_type": curr_cols[col]["type"],
                    "severity": "INFO",
                })

        for col in base_cols:
            if col not in curr_cols:
                continue
            if base_cols[col]["type"] != curr_cols[col]["type"]:
                changes.append({
                    "change": "TYPE_CHANGED",
                    "table": table,
                    "column": col,
                    "was_type": base_cols[col]["type"],
                    "now_type": curr_cols[col]["type"],
                    "severity": "HIGH",
                })

    return changes


def _checksum(schema: dict) -> str:
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]


def _print_drift_report(changes: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("⚠️  SCHEMA DRIFT DETECTED")
    print("=" * 60)
    for c in changes:
        sev = c.get("severity", "UNKNOWN")
        symbol = "🔴" if sev == "CRITICAL" else "🟡" if sev == "HIGH" else "🟢"
        print(f"{symbol} [{sev}] {c['change']} — {c.get('table')}.{c.get('column', '')}")
        if "impact" in c:
            print(f"   Impact: {c['impact']}")
    print("=" * 60 + "\n")


async def _fire_alert(drift_report: dict) -> None:
    """Send alert to configured webhook (Slack, PagerDuty, etc.)."""
    if not settings.SCHEMA_ALERT_WEBHOOK:
        return
    payload = {
        "text": f"⚠️ *Schema drift detected* in {settings.SQLITE_PATH}",
        "attachments": [
            {
                "color": "danger",
                "text": json.dumps(drift_report["changes"], indent=2),
                "title": f"{len(drift_report['changes'])} schema changes since last snapshot",
            }
        ],
    }
    async with httpx.AsyncClient() as client:
        await client.post(settings.SCHEMA_ALERT_WEBHOOK, json=payload, timeout=5)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import sys
    import typer

    app = typer.Typer()

    @app.command()
    def snapshot(label: str = "pre_migration"):
        """Take a schema snapshot. Run BEFORE migrations."""
        asyncio.run(take_snapshot(label=label))

    @app.command()
    def diff(baseline: str | None = None):
        """Detect schema drift vs most recent snapshot."""
        result = asyncio.run(detect_drift(baseline_path=baseline))
        if result["drifted"]:
            typer.echo(f"DRIFT DETECTED: {len(result['changes'])} changes")
            raise typer.Exit(code=1)   # Non-zero exit blocks CI deploy
        else:
            typer.echo("✓ No schema drift detected")

    app()
