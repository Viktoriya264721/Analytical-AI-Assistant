import pandas as pd
from typing import Set, Optional, Dict
from supabase import Client

from analytics.queries import fetch_all_tables, get_all_months
from analytics.pnl import compute_pnl
from analytics.cashflow import compute_cashflow
from analytics.operational import (
    compute_efficiency,
    compute_service_counts,
    compute_salary_details,
    compute_revenue_per_person,
    compute_service_type_stats,
)
from analytics.metrics_writer import upsert_metrics


def _compute_month(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> list:
    """Computes ALL metrics for one month."""
    metrics = []

    pnl = compute_pnl(tables, month)
    metrics.extend(pnl)

    cf = compute_cashflow(tables, month, pnl)
    metrics.extend(cf)

    metrics.extend(compute_efficiency(tables, month))
    metrics.extend(compute_service_counts(tables, month))
    metrics.extend(compute_salary_details(tables, month))
    metrics.extend(compute_revenue_per_person(tables, month))
    metrics.extend(compute_service_type_stats(tables, month))

    return metrics


def compute_monthly_metrics(
    supabase: Client,
    months: Optional[Set[str]] = None,
):
    """
    Computes and writes monthly metrics.

    months=None  → recomputes ALL months from the database.
    months={...} → only the specified months.
    """
    print("\nComputing monthly metrics...")

    tables = fetch_all_tables(supabase)
    all_months = get_all_months(tables)

    if not all_months:
        print("  No data found in database.")
        return

    if months:
        target_months = sorted(pd.to_datetime(list(months)))
    else:
        target_months = all_months

    print(f"  Months to compute: {len(target_months)}")

    all_metrics = []

    for month in target_months:
        month_metrics = _compute_month(tables, month)
        all_metrics.extend(month_metrics)
        print(f"    {month.strftime('%Y-%m')}: {len(month_metrics)} metrics")

    if all_metrics:
        count = upsert_metrics(supabase, all_metrics)
        print(f"  Written {count} metrics to monthly_metrics")
    else:
        print("  No metrics to write.")


def compute_for_updated_months(
    supabase: Client,
    touched_months: Set[str],
):
    """
    Computes metrics only for months that were updated during sync.
    Called automatically after the sync pipeline.
    """
    if not touched_months:
        print("  No updated months to compute.")
        return

    compute_monthly_metrics(supabase, touched_months)
