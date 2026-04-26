"""Raw DataFrame cleaning: column normalisation, type coercion, and date parsing."""

import pandas as pd
from typing import Dict


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert column names to lowercase snake_case."""
    df = df.copy()
    df.columns = (
        df.columns
        .map(lambda x: str(x).strip().lower())
        .map(lambda x: x.replace(" ", "_"))
        .map(lambda x: x.replace("-", "_"))
    )
    return df


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns and replace empty strings with NA."""
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA})
        )

    return df


def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ``дата`` → ``date`` (datetime64) and drop the original; no-op when absent."""
    df = df.copy()

    if "дата" not in df.columns:
        return df

    df["date"] = pd.to_datetime(
        df["дата"],
        errors="coerce",
        dayfirst=True
    )

    df = df.drop(columns=["дата"])
    return df


def _parse_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns to numeric where possible, handling comma-decimal notation."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            try:
                converted = pd.to_numeric(
                    df[col].str.replace(",", ".", regex=False),
                    errors="raise"
                )
                df[col] = converted
            except (ValueError, TypeError):
                pass

    return df


def clean_tables(raw_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Apply the full cleaning pipeline to all ingested tables.

    Steps applied per table:
    1. Normalise column names to lowercase snake_case.
    2. Strip strings and replace empty values with NA.
    3. Parse the ``дата`` column into a ``date`` datetime column.
    4. Coerce numeric-looking object columns to float/int.

    Args:
        raw_tables: Raw DataFrames from the ingestion layer.

    Returns:
        Cleaned DataFrames ready for normalisation.
    """
    cleaned: Dict[str, pd.DataFrame] = {}

    for table_name, df in raw_tables.items():
        if df.empty:
            cleaned[table_name] = df
            continue

        tmp = df.copy()
        tmp = _normalize_columns(tmp)
        tmp = _clean_strings(tmp)
        tmp = _parse_date(tmp)
        tmp = _parse_numbers(tmp)

        cleaned[table_name] = tmp

    return cleaned
