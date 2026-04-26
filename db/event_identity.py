import hashlib
import pandas as pd

from config.event_registry import get_uid_fields, get_content_fields
from config.settings import CANONICAL_DATE_FORMAT


def _normalize_date(value) -> str:
    """Stable date format for hashing (YYYY-MM-DD)."""
    if pd.isna(value):
        return "NULL"
    if isinstance(value, pd.Timestamp):
        return value.strftime(CANONICAL_DATE_FORMAT)
    if hasattr(value, "strftime"):
        return value.strftime(CANONICAL_DATE_FORMAT)
    try:
        return pd.Timestamp(value).strftime(CANONICAL_DATE_FORMAT)
    except Exception:
        return str(value)


def generate_source_uid(row: pd.Series, table_name: str) -> str:
    """
    Stable row identifier.
    Does NOT depend on changeable values (amounts, hours, etc.).
    Fields per table are defined in TABLE_REGISTRY.
    """
    fields = get_uid_fields(table_name)

    parts = []
    for f in fields:
        value = row.get(f)
        if f == "date":
            parts.append(_normalize_date(value))
        elif pd.isna(value):
            parts.append("NULL")
        else:
            parts.append(str(value))

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_content_hash(row: pd.Series, table_name: str) -> str:
    """
    Fingerprint of changeable row values.
    If it changes — the row is updated on upsert.
    """
    fields = get_content_fields(table_name)

    parts = []
    for f in fields:
        value = row.get(f)
        if pd.isna(value):
            value = "NULL"
        parts.append(f"{f}={value}")

    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
