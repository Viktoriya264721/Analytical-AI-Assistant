import hashlib
from typing import List, Dict
import numpy as np
import pandas as pd
from supabase import Client


BATCH_SIZE = 500


def _to_python(v):
    """Convert a numpy/pandas scalar to a JSON-serialisable Python native."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def generate_metric_uid(month, metric_name: str, person, category=None) -> str:
    """Generate a stable unique identifier for a metric row.

    SHA-256 hash of: month | metric_name | person_or_NULL | category_or_NULL.
    The category component distinguishes metrics that share the same name but
    differ by service type (e.g. service_units split by activity type).
    """
    month_str = pd.Timestamp(month).strftime("%Y-%m-%d")
    person_str = str(person) if person and not pd.isna(person) else "NULL"
    category_str = str(category) if category and not pd.isna(category) else "NULL"
    raw = f"{month_str}|{metric_name}|{person_str}|{category_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _serialize_metric(metric: dict) -> dict:
    """Prepare a metric record for JSON serialisation and upsert."""
    month = metric["month"]
    if isinstance(month, pd.Timestamp):
        month = month.date().isoformat()
    elif hasattr(month, "isoformat"):
        month = month.isoformat()

    person = metric.get("person")
    if person is not None and pd.isna(person):
        person = None

    category = metric.get("category")

    metric_uid = generate_metric_uid(
        metric["month"], metric["metric_name"], person, category
    )

    return {
        "metric_uid": metric_uid,
        "month": str(month),
        "metric_name": metric["metric_name"],
        "metric_value": _to_python(metric["metric_value"]),
        "person": person,
        "category": metric["category"],
    }


def upsert_metrics(supabase: Client, metrics: List[Dict]):
    """Batch-upsert metrics into monthly_metrics.

    New rows are inserted; changed rows are updated; unchanged rows are skipped.
    Returns the total number of records written.
    """
    if not metrics:
        return 0

    records = [_serialize_metric(m) for m in metrics]

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i : i + BATCH_SIZE]
        supabase.table("monthly_metrics").upsert(
            batch,
            on_conflict="metric_uid",
        ).execute()

    return len(records)
