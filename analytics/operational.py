"""Per-person and per-activity operational metrics for a given month."""

from typing import List, Dict
import pandas as pd

from analytics.queries import filter_month


def compute_efficiency(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> List[Dict]:
    """
    Efficiency per rehab employee:
    efficiency_pct = (SUM(units) / SUM(available_hours)) * 100

    units - from specialist_activity table
    available_hours - from specialist_capacity table
    """
    activity_m = filter_month(tables.get("specialist_activity", pd.DataFrame()), month)
    capacity_m = filter_month(tables.get("specialist_capacity", pd.DataFrame()), month)

    actual = (
        activity_m.groupby("person")["units"].sum().to_dict()
        if not activity_m.empty else {}
    )
    max_hours = (
        capacity_m.groupby("person")["available_hours"].sum().to_dict()
        if not capacity_m.empty else {}
    )

    metrics = []
    all_persons = set(actual.keys()) | set(max_hours.keys())

    for person in sorted(all_persons):
        act = actual.get(person, 0)
        mx = max_hours.get(person, 0)
        eff = round((act / mx) * 100, 1) if mx and mx > 0 else 0.0

        metrics.append({
            "month": month,
            "metric_name": "efficiency_pct",
            "metric_value": eff,
            "person": person,
            "category": "efficiency",
        })

    return metrics


def compute_service_counts(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> List[Dict]:
    """Session counts per employee + total."""
    activity_m = filter_month(tables.get("specialist_activity", pd.DataFrame()), month)
    metrics = []

    if activity_m.empty:
        metrics.append({
            "month": month, "metric_name": "total_services",
            "metric_value": 0, "person": None, "category": "operations",
        })
        return metrics

    by_person = activity_m.groupby("person")["units"].sum()

    for person, count in by_person.items():
        metrics.append({
            "month": month, "metric_name": "service_count",
            "metric_value": round(float(count), 2), "person": person,
            "category": "operations",
        })

    metrics.append({
        "month": month, "metric_name": "total_services",
        "metric_value": round(float(by_person.sum()), 2), "person": None,
        "category": "operations",
    })

    return metrics


def compute_revenue_per_person(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> List[Dict]:
    """
    Revenue brought by each specialist for the month.
    Reads generated_revenue from specialist_payouts table.
    """
    payouts_m = filter_month(tables.get("specialist_payouts", pd.DataFrame()), month)
    metrics = []

    if payouts_m.empty or "generated_revenue" not in payouts_m.columns:
        return metrics

    rows = payouts_m[payouts_m["generated_revenue"].notna()]

    for _, row in rows.iterrows():
        metrics.append({
            "month":        month,
            "metric_name":  "person_income",
            "metric_value": round(float(row["generated_revenue"]), 2),
            "person":       row["person"],
            "category":     "revenue",
        })

    return metrics


def compute_service_type_stats(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> List[Dict]:
    """
    Total units per activity type for the month.
    metric_name = "service_units", category = activity_type.
    """
    activity_m = filter_month(tables.get("specialist_activity", pd.DataFrame()), month)

    if activity_m.empty or "activity_type" not in activity_m.columns:
        return []

    by_type = activity_m.groupby("activity_type")["units"].sum()

    return [
        {
            "month":        month,
            "metric_name":  "service_units",
            "metric_value": round(float(count), 2),
            "person":       None,
            "category":     activity_type,
        }
        for activity_type, count in by_type.items()
    ]


def compute_salary_details(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> List[Dict]:
    """Payout amount per employee for the month."""
    payouts_m = filter_month(tables.get("specialist_payouts", pd.DataFrame()), month)
    metrics = []

    if payouts_m.empty:
        return metrics

    by_person = payouts_m.groupby("person")["payout_amount"].sum()

    for person, amount in by_person.items():
        metrics.append({
            "month": month, "metric_name": "salary_amount",
            "metric_value": round(float(amount), 2), "person": person,
            "category": "operations",
        })

    return metrics
