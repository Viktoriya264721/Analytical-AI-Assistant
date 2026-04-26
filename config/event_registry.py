
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd


@dataclass(frozen=True)
class TableSpec:
    """Schema descriptor for a single domain table.

    Attributes:
        table_name: Table name in Supabase.
        uid_fields: Fields used to compute the stable ``source_uid`` hash.
        content_fields: Fields used to compute ``content_hash`` for change detection.
        required_fields: Fields that must not be NULL (validated before upsert).
        columns: All domain columns, excluding ``source_uid``, ``content_hash``,
            and timestamp columns.
        has_person: Whether the table contains a ``person`` column.
    """
    table_name: str
    uid_fields: tuple
    content_fields: tuple
    required_fields: frozenset
    columns: tuple
    has_person: bool = False


TABLE_REGISTRY: Dict[str, TableSpec] = {}


def _register(spec: TableSpec):
    TABLE_REGISTRY[spec.table_name] = spec


_register(TableSpec(
    table_name="daily_revenue",
    uid_fields=("date",),
    content_fields=("total_revenue", "card_revenue", "cash_revenue"),
    required_fields=frozenset({"date", "total_revenue"}),
    columns=("date", "total_revenue", "card_revenue", "cash_revenue"),
    has_person=False,
))

_register(TableSpec(
    table_name="expenses",
    uid_fields=("date", "category"),
    content_fields=("amount",),
    required_fields=frozenset({"date", "category", "amount"}),
    columns=("date", "category", "amount"),
    has_person=False,
))

_register(TableSpec(
    table_name="amortization",
    uid_fields=("date", "asset_name"),
    content_fields=("total_amount", "duration_months"),
    required_fields=frozenset({"date", "asset_name", "total_amount", "duration_months"}),
    columns=("date", "asset_name", "total_amount", "duration_months"),
    has_person=False,
))

_register(TableSpec(
    table_name="specialist_capacity",
    uid_fields=("date", "person"),
    content_fields=("available_hours",),
    required_fields=frozenset({"date", "person", "available_hours"}),
    columns=("date", "person", "available_hours"),
    has_person=True,
))

_register(TableSpec(
    table_name="specialist_activity",
    uid_fields=("date", "person", "activity_type"),
    content_fields=("units",),
    required_fields=frozenset({"date", "person", "activity_type"}),
    columns=("date", "person", "units", "activity_type"),
    has_person=True,
))

_register(TableSpec(
    table_name="specialist_payouts",
    uid_fields=("date", "person"),
    content_fields=("payout_amount", "generated_revenue"),
    required_fields=frozenset({"date", "person", "payout_amount"}),
    columns=("date", "person", "payout_amount", "generated_revenue"),
    has_person=True,
))


def get_spec(table_name: str) -> TableSpec:
    """Return the :class:`TableSpec` for *table_name*.

    Raises:
        ValueError: When *table_name* is not registered.
    """
    if table_name not in TABLE_REGISTRY:
        raise ValueError(
            f"Unknown table='{table_name}'. "
            f"Known tables: {sorted(TABLE_REGISTRY.keys())}"
        )
    return TABLE_REGISTRY[table_name]


def get_uid_fields(table_name: str) -> tuple:
    """Return the uid field tuple for *table_name*."""
    return get_spec(table_name).uid_fields


def get_content_fields(table_name: str) -> tuple:
    """Return the content hash field tuple for *table_name*."""
    return get_spec(table_name).content_fields


def get_required_fields(table_name: str) -> frozenset:
    """Return the required fields frozenset for *table_name*."""
    return get_spec(table_name).required_fields


def filter_to_table_columns(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Drop any columns not declared in the table schema.

    Args:
        df: Input DataFrame.
        table_name: Registered table name.

    Returns:
        DataFrame containing only the declared domain columns that are
        present in *df*.
    """
    spec = get_spec(table_name)
    keep = [c for c in spec.columns if c in df.columns]
    return df[keep]
