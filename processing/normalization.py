"""Per-table normalisation: column mapping, type enforcement, and schema filtering."""

import pandas as pd
from typing import Dict

from config.event_registry import filter_to_table_columns


_TABLE_TYPES: dict[str, dict[str, str]] = {
    "daily_revenue": {
        "date": "datetime64[ns]",
        "total_revenue": "float64",
        "card_revenue": "float64",
        "cash_revenue": "float64",
    },
    "expenses": {
        "date": "datetime64[ns]",
        "amount": "float64",
    },
    "amortization": {
        "date": "datetime64[ns]",
        "total_amount": "float64",
        "duration_months": "int64",
    },
    "specialist_capacity": {
        "date": "datetime64[ns]",
        "available_hours": "float64",
    },
    "specialist_activity": {
        "date": "datetime64[ns]",
        "units": "int64",
    },
    "specialist_payouts": {
        "date": "datetime64[ns]",
        "payout_amount": "float64",
        "generated_revenue": "float64",
    },
}


def _normalize_text(x) -> str:
    """Strip and lowercase a text value; preserve NA as-is."""
    if pd.isna(x):
        return pd.NA
    return str(x).strip().lower()


def _apply_types(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Cast columns to the types declared in ``_TABLE_TYPES``."""
    schema = _TABLE_TYPES.get(table_name, {})
    for col, dtype in schema.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "float64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            elif dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        except Exception:
            pass
    return df


def _finalize(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Filter to schema-declared columns and enforce their types."""
    if df.empty:
        return df
    df = filter_to_table_columns(df, table_name)
    df = _apply_types(df, table_name)
    return df


def normalize_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``daily_revenue`` table."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        records.append({
            "date":          row["date"],
            "total_revenue": row.get("total_revenue"),
            "card_revenue":  row.get("card_revenue"),
            "cash_revenue":  row.get("cash_revenue"),
        })

    return _finalize(pd.DataFrame(records), "daily_revenue")


def normalize_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``expenses`` table."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        records.append({
            "date":     row["date"],
            "category": _normalize_text(row.get("category")),
            "amount":   row.get("amount"),
        })

    return _finalize(pd.DataFrame(records), "expenses")


def normalize_amortization(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``amortization`` table."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        records.append({
            "date":             row["date"],
            "asset_name":       _normalize_text(row.get("asset_name")),
            "total_amount":     row.get("total_amount"),
            "duration_months":  row.get("duration_months"),
        })

    return _finalize(pd.DataFrame(records), "amortization")


def normalize_specialist_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``specialist_capacity`` table."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        records.append({
            "date":            row["date"],
            "person":          _normalize_text(row.get("employee")),
            "available_hours": row.get("available_hours"),
        })

    return _finalize(pd.DataFrame(records), "specialist_capacity")


def normalize_specialist_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``specialist_activity`` table."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        records.append({
            "date":          row["date"],
            "person":        _normalize_text(row.get("specialist")),
            "units":         row.get("units"),
            "activity_type": _normalize_text(row.get("activity_type")),
        })

    return _finalize(pd.DataFrame(records), "specialist_activity")


def normalize_specialist_payouts(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the ``specialist_payouts`` table."""
    if df.empty:
        return df

    records = []
    for _, row in df.iterrows():
        gen_rev = row.get("generated_revenue")
        records.append({
            "date":              row["date"],
            "person":            _normalize_text(row.get("specialist")),
            "payout_amount":     row.get("payout_amount"),
            "generated_revenue": None if pd.isna(gen_rev) else gen_rev,
        })

    return _finalize(pd.DataFrame(records), "specialist_payouts")


NORMALIZER_MAP = {
    "daily_revenue":       normalize_daily_revenue,
    "expenses":            normalize_expenses,
    "amortization":        normalize_amortization,
    "specialist_capacity": normalize_specialist_capacity,
    "specialist_activity": normalize_specialist_activity,
    "specialist_payouts":  normalize_specialist_payouts,
}


def normalize_tables(
    cleaned_tables: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Apply per-table normalisation to all cleaned tables.

    Tables without a registered normaliser are skipped with a warning.

    Args:
        cleaned_tables: Output of the cleaning pipeline.

    Returns:
        Normalised DataFrames keyed by table name.
    """
    normalized: Dict[str, pd.DataFrame] = {}

    for sheet_name, df in cleaned_tables.items():
        normalizer = NORMALIZER_MAP.get(sheet_name)
        if normalizer is None:
            print(f"  No normalizer for sheet '{sheet_name}', skipping")
            continue

        normalized[sheet_name] = normalizer(df)

    return normalized
