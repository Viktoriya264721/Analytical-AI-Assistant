from typing import Dict, List, Set
import numpy as np
import pandas as pd
from supabase import Client

from config.event_registry import TABLE_REGISTRY


BATCH_SIZE = 500


def _to_python(v):
    """Convert numpy/pandas types to Python native for JSON serialization."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _serialize_record(record: dict, allowed_columns: set) -> dict:
    """
    Prepares a row for JSON:
    - Timestamp → ISO date string
    - NaN/NA → None
    - numpy types → Python native
    - Filters to allowed columns only
    """
    clean = {}
    for k, v in record.items():
        if k not in allowed_columns:
            continue
        if pd.isna(v):
            clean[k] = None
        elif isinstance(v, pd.Timestamp):
            clean[k] = v.date().isoformat()
        else:
            clean[k] = _to_python(v)
    return clean


def upsert_table(
    supabase: Client,
    table_name: str,
    df: pd.DataFrame,
) -> int:
    """
    Upserts one table to Supabase.
    Conflict key: source_uid — updates existing rows, inserts new ones.
    Returns number of rows written.
    """
    if df.empty:
        return 0

    spec = TABLE_REGISTRY[table_name]
    allowed = set(spec.columns) | {"source_uid", "content_hash"}

    records = [
        _serialize_record(r, allowed)
        for r in df.to_dict(orient="records")
    ]

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        supabase.table(table_name).upsert(
            batch,
            on_conflict="source_uid",
        ).execute()

    return len(records)


def upsert_all(
    supabase: Client,
    tables: Dict[str, pd.DataFrame],
) -> Set[str]:
    """
    Writes all normalized tables to Supabase.
    Skips tables not registered in TABLE_REGISTRY.
    Returns set of months that were updated (for metrics pipeline).
    """
    grand_total = 0
    touched_months: Set[str] = set()

    for table_name, df in tables.items():
        if table_name not in TABLE_REGISTRY:
            print(f"  [{table_name}] not in registry, skipping")
            continue

        count = upsert_table(supabase, table_name, df)
        grand_total += count
        status = f"{count} rows" if count > 0 else "empty, skipped"
        print(f"  {table_name:<20} -> {status}")

        if not df.empty and "date" in df.columns:
            months = (
                pd.to_datetime(df["date"], errors="coerce")
                .dt.to_period("M")
                .dt.to_timestamp()
                .dropna()
                .unique()
            )
            for m in months:
                touched_months.add(m.strftime("%Y-%m-%d"))

    print(f"  {'TOTAL':<20} -> {grand_total} rows written")
    return touched_months
