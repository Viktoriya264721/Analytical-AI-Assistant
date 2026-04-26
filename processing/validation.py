"""Pre-upsert validation: required columns and null checks."""

import pandas as pd
from typing import Dict

from config.event_registry import TABLE_REGISTRY, get_required_fields


def validate_table(df: pd.DataFrame, table_name: str) -> None:
    """Validate a normalised DataFrame against its registered schema.

    Checks:
    1. The table is registered in ``TABLE_REGISTRY``.
    2. All required columns are present.
    3. No NaN values exist in required columns.

    Args:
        df: Normalised DataFrame to validate.
        table_name: Registered table name.

    Raises:
        ValueError: On any schema or data quality violation.
    """
    if df.empty:
        return

    if table_name not in TABLE_REGISTRY:
        raise ValueError(
            f"Unknown table '{table_name}'. "
            f"Register it in config/event_registry.py"
        )

    required = get_required_fields(table_name)

    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"[{table_name}] Missing required columns: {missing_cols}"
        )

    for col in required:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            raise ValueError(
                f"[{table_name}] {nan_count} NaN values in required column '{col}'"
            )


def validate_all(normalized_tables: Dict[str, pd.DataFrame]) -> None:
    """Validate all normalised tables.

    Call after normalisation and before identity attachment.

    Args:
        normalized_tables: Output of the normalisation pipeline.

    Raises:
        ValueError: On the first validation failure encountered.
    """
    for table_name, df in normalized_tables.items():
        validate_table(df, table_name)
