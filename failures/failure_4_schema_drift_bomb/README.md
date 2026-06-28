# Failure 4: The Schema Drift Bomb

> **Production symptom:** The agent returned "Supplier not found" for every supplier query.
> It had worked perfectly for 3 weeks. No code was deployed. No config changed.
> The engineering team spent 2 days assuming it was a data quality issue before
> anyone checked the database schema.

---

## Timeline

```
Week 1–3:  Agent works perfectly. 1,200+ supplier lookups, 0 failures.

Monday:    Database team runs quarterly schema normalization migration.
           Renames suppliers.supplier_id → suppliers.supplier_code
           "It's just a rename," they said. "Nothing queries that directly."

Tuesday:   Agent returns "Supplier not found" for all queries.
           Support ticket opened: "Data quality issue in SAP mirror?"

Wednesday: Data team investigates. Data is fine — all 50 suppliers present.
           AI team investigates. Retrieval is fine — contracts found correctly.
           Nobody checks the schema.

Thursday:  Junior engineer runs: SELECT * FROM suppliers LIMIT 1;
           Sees: supplier_code, not supplier_id.
           Root cause found. Schema migration renamed the column.
           2 days of debugging for a 30-second fix.
```

---

## The Wrong Diagnosis

- "SAP data sync job failed" — investigated for 8 hours, data was fine
- "Redis cache is stale" — cache invalidated, didn't help
- "Qdrant vectors are stale" — re-ingested, didn't help
- "The LLM is hallucinating 'not found'" — tested without LLM, same result

Every investigation looked at the AI stack. Nobody looked at the database schema.
That's the insidious thing about schema drift: it looks like a data problem,
not a schema problem.

---

## Actual Root Cause

```python
# src/data/database.py — the query that broke
async with db.execute(
    "SELECT * FROM suppliers WHERE supplier_id = ?",  # Column renamed!
    (supplier_id,),
) as cursor:
    row = await cursor.fetchone()
    return dict(row) if row else None  # Returns None for ALL suppliers
```

SQLite silently returns empty results for `WHERE renamed_column = ?` instead of
raising an error. The migration team renamed the column, the AI team's query
broke silently, and for 2 days everyone looked everywhere except the schema.

---

## The Fix

**Schema drift detection pipeline with CI gate.**

### How it works

```bash
# BEFORE a migration (run by database team):
make schema-snapshot
# → Saves data/schema_snapshots/schema_20240315T140000Z_pre_migration.json

# Migration runs...

# In CI pipeline, automatically after migration:
make schema-diff
# → Compares current schema against most recent snapshot

# Output:
# ⚠️  SCHEMA DRIFT DETECTED
# 🔴 [CRITICAL] COLUMN_DROPPED — suppliers.supplier_id
#    Impact: Any query referencing suppliers.supplier_id will now fail
# 🟢 [INFO] COLUMN_ADDED — suppliers.supplier_code
```

### CI gate (GitHub Actions)

```yaml
# .github/workflows/eval-gate.yml
- name: Schema drift check
  run: python -m src.data.schema_monitor diff
  # Exit code 1 if drift detected — blocks deploy
```

The deploy is blocked until either:
1. The AI team updates their queries to use the new column name, OR
2. The migration is rolled back

### Alert webhook

If `SCHEMA_ALERT_WEBHOOK` is set, a Slack/PagerDuty alert fires immediately
when drift is detected — even before the CI pipeline completes.

---

## The Schema Snapshot Format

```json
{
  "captured_at": "2024-03-15T14:00:00Z",
  "label": "pre_migration",
  "schema": {
    "suppliers": {
      "supplier_id": {
        "type": "TEXT",
        "notnull": true,
        "primary_key": true
      }
    }
  },
  "checksum": "a3f9b2c1d4e5"
}
```

---

## Demo: Trigger the failure

```bash
# 1. Seed database (creates suppliers.supplier_id)
make seed

# 2. Take baseline snapshot
make schema-snapshot

# 3. Run migration that renames the column
python failures/failure_4_schema_drift_bomb/trigger_migration.py

# 4. Detect drift (would have blocked deploy)
make schema-diff
# Output: DRIFT DETECTED — 2 changes
#   🔴 COLUMN_DROPPED: suppliers.supplier_id
#   🟢 COLUMN_ADDED: suppliers.supplier_code
```

---

## Before / After

| | Before | After |
|--|--------|-------|
| Time to detect schema drift | 2 days (manual) | < 60 seconds (automated) |
| Deploy blocked on drift? | No | Yes (CI gate) |
| Alert on drift? | No | Yes (webhook) |
| MTTR for this class of failure | 2+ days | Minutes |
