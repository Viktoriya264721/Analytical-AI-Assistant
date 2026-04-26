import pandas as pd
from typing import Dict, Set
from supabase import Client


PAGE_SIZE = 1000

DOMAIN_TABLES = [
    "daily_revenue",
    "expenses",
    "amortization",
    "specialist_capacity",
    "specialist_activity",
    "specialist_payouts",
]


def _fetch_table(supabase: Client, table_name: str) -> pd.DataFrame:
    """Paginated fetch of all rows from a single table."""
    all_data = []
    offset = 0

    while True:
        response = (
            supabase.table(table_name)
            .select("*")
            .range(offset, offset + PAGE_SIZE - 1)
            .execute()
        )

        if not response.data:
            break

        all_data.extend(response.data)

        if len(response.data) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


def fetch_all_tables(supabase: Client) -> Dict[str, pd.DataFrame]:
    """
    Loads all domain tables from Supabase.
    Returns dict {table_name: DataFrame}.
    """
    tables = {}
    total_rows = 0

    for name in DOMAIN_TABLES:
        df = _fetch_table(supabase, name)
        tables[name] = df
        total_rows += len(df)
        print(f"  {name:<20} -> {len(df)} rows")

    print(f"  {'TOTAL':<20} -> {total_rows} rows fetched")
    return tables


def filter_month(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Filters a DataFrame to rows belonging to the given month."""
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    return df[
        df["date"].dt.to_period("M").dt.to_timestamp() == month
    ].copy()


def get_all_months(tables: Dict[str, pd.DataFrame]) -> list:
    """Returns sorted list of unique months across all tables."""
    months: Set[pd.Timestamp] = set()

    for df in tables.values():
        if df.empty or "date" not in df.columns:
            continue
        for m in df["date"].dt.to_period("M").dt.to_timestamp().dropna().unique():
            months.add(m)

    return sorted(months)


def get_rehab_persons(tables: Dict[str, pd.DataFrame]) -> set:
    """
    Identifies rehab employees: persons appearing in specialist_activity or
    specialist_capacity tables.
    """
    persons = set()

    for table_name in ("specialist_activity", "specialist_capacity"):
        df = tables.get(table_name, pd.DataFrame())
        if not df.empty and "person" in df.columns:
            persons.update(df["person"].dropna().unique())

    return persons
