from __future__ import annotations

import numpy as np
import pandas as pd
from supabase import Client

_PAGE_SIZE = 1000


def _to_python(value: object) -> object:
    """Convert numpy/pandas scalars to JSON-serialisable Python natives."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return 0
    return value


def _fetch_all_metrics(supabase: Client) -> pd.DataFrame:
    """Fetch all rows from monthly_metrics.

    Returns:
        DataFrame with a parsed datetime ``month`` column, or an empty
        DataFrame when the table contains no rows.
    """
    rows: list[dict] = []
    offset = 0

    while True:
        response = (
            supabase.table("monthly_metrics")
            .select("*")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        if not response.data:
            break
        rows.extend(response.data)
        if len(response.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df



def _fetch_expense_rows(supabase: Client) -> pd.DataFrame:
    """Fetch all rows from the expenses domain table.

    Returns:
        DataFrame with columns ``date``, ``amount``, ``category``, or an
        empty DataFrame when there are no expense records.
    """
    rows: list[dict] = []
    offset = 0

    while True:
        response = (
            supabase.table("expenses")
            .select("date, amount, category")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        if not response.data:
            break
        rows.extend(response.data)
        if len(response.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _metrics_for_month(
    metrics_df: pd.DataFrame,
    target: pd.Timestamp,
) -> pd.DataFrame:
    """Filter metrics_df to rows belonging to *target* month."""
    return metrics_df[
        metrics_df["month"].dt.to_period("M").dt.to_timestamp() == target
    ]


def _scalar_dict(month_df: pd.DataFrame, category: str) -> dict[str, float]:
    """Extract {metric_name: metric_value} for aggregate (non-person) metrics."""
    subset = month_df[
        (month_df["category"] == category) & month_df["person"].isna()
    ]
    return {
        row["metric_name"]: _to_python(round(row["metric_value"], 2))
        for _, row in subset.iterrows()
    }


def _build_salaries(month_df: pd.DataFrame) -> dict:
    """Build salary summary: total and top-3 highest-paid employees."""
    salary_rows = month_df[
        (month_df["metric_name"] == "salary_amount") & month_df["person"].notna()
    ]
    if salary_rows.empty:
        return {"total": 0, "top_3": {}}

    total = _to_python(round(salary_rows["metric_value"].sum(), 2))
    top_3 = (
        salary_rows.nlargest(3, "metric_value")
        .set_index("person")["metric_value"]
        .round(2)
        .apply(_to_python)
        .to_dict()
    )
    return {"total": total, "top_3": top_3}


def _build_services(month_df: pd.DataFrame) -> dict:
    """Build service-count summary: total units and per-employee breakdown."""
    total_row = month_df[month_df["metric_name"] == "total_services"]
    total = (
        _to_python(round(total_row["metric_value"].sum(), 2))
        if not total_row.empty
        else 0
    )

    per_person = month_df[
        (month_df["metric_name"] == "service_count") & month_df["person"].notna()
    ]
    by_employee = (
        per_person.set_index("person")["metric_value"]
        .round(2)
        .sort_values(ascending=False)
        .apply(_to_python)
        .to_dict()
        if not per_person.empty
        else {}
    )
    return {"total": total, "by_employee": by_employee}


def _build_efficiency(month_df: pd.DataFrame) -> dict[str, float]:
    """Build per-person efficiency percentage mapping."""
    eff_rows = month_df[
        (month_df["metric_name"] == "efficiency_pct") & month_df["person"].notna()
    ]
    if eff_rows.empty:
        return {}
    return (
        eff_rows.set_index("person")["metric_value"]
        .round(1)
        .apply(_to_python)
        .to_dict()
    )


def _build_center_load(month_df: pd.DataFrame) -> dict:
    """Extract center-load metrics: gaps, overloads, utilisation."""
    def _get(name: str) -> float:
        rows = month_df[month_df["metric_name"] == name]
        return _to_python(round(rows["metric_value"].iloc[0], 1)) if not rows.empty else 0

    return {
        "gaps": _get("gap_hours"),
        "overloads": _get("overload_hours"),
        "utilization_pct": _get("center_utilization_pct"),
    }


def _fetch_daily_services(supabase: Client, target: pd.Timestamp) -> dict:
    """Return daily service units for the target month from specialist_activity."""
    month_start = target.strftime("%Y-%m-01")
    month_end   = (target + pd.DateOffset(months=1)).strftime("%Y-%m-01")

    rows: list[dict] = []
    offset = 0
    while True:
        response = (
            supabase.table("specialist_activity")
            .select("date, units")
            .gte("date", month_start)
            .lt("date", month_end)
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        if not response.data:
            break
        rows.extend(response.data)
        if len(response.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    if not rows:
        return {"dates": [], "units": []}

    df = pd.DataFrame(rows)
    df["date"]  = pd.to_datetime(df["date"], errors="coerce")
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0)

    daily = df.groupby("date")["units"].sum().sort_index()
    return {
        "dates": [d.strftime("%d.%m") for d in daily.index],
        "units": [int(v) for v in daily.values],
    }


def _fetch_name_map(supabase: Client) -> dict[str, str]:
    """Return {anonymous_id: real_name} mapping from the persons table."""
    rows: list[dict] = []
    offset = 0

    while True:
        response = (
            supabase.table("persons")
            .select("anonymous_id, real_name")
            .range(offset, offset + _PAGE_SIZE - 1)
            .execute()
        )
        if not response.data:
            break
        rows.extend(response.data)
        if len(response.data) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    return {
        row["anonymous_id"]: row["real_name"]
        for row in rows
        if row.get("anonymous_id") and row.get("real_name")
    }


def _apply_names(d: dict, name_map: dict[str, str]) -> dict:
    """Replace anonymous_id keys with real names where mapping exists."""
    return {name_map.get(k, k): v for k, v in d.items()}


def _build_service_types(month_df: pd.DataFrame) -> dict[str, float]:
    """Build service type breakdown (масаж, лфк, фізіотерапія)."""
    type_rows = month_df[month_df["metric_name"] == "service_units"]
    if type_rows.empty:
        return {}
    return (
        type_rows.set_index("category")["metric_value"]
        .round(1)
        .apply(_to_python)
        .to_dict()
    )


def _build_expense_detail(
    expense_df: pd.DataFrame,
    target: pd.Timestamp,
) -> dict:
    """Summarise expenses for *target* month: total and top-3 categories."""
    if expense_df.empty:
        return {"total": 0, "top_3": {}}

    month_exp = expense_df[
        expense_df["date"].dt.to_period("M").dt.to_timestamp() == target
    ]
    if month_exp.empty:
        return {"total": 0, "top_3": {}}

    total = _to_python(round(month_exp["amount"].sum(), 2))
    top_3 = (
        month_exp.groupby("category")["amount"]
        .sum()
        .round(2)
        .sort_values(ascending=False)
        .head(3)
        .apply(_to_python)
        .to_dict()
    )
    return {"total": total, "top_3": top_3}


def _build_history(
    all_metrics: pd.DataFrame,
    target: pd.Timestamp,
) -> dict:
    """Build historical P&L time-series for months preceding *target*."""
    pnl = all_metrics[
        (all_metrics["category"] == "pnl")
        & all_metrics["person"].isna()
        & (all_metrics["month"] < target)
    ]
    if pnl.empty:
        return {"months": [], "revenue": [], "net_profit": [], "ebit": []}

    months_sorted = sorted(pnl["month"].unique())

    def _series(metric_name: str) -> list[float]:
        result = []
        for m in months_sorted:
            row = pnl[(pnl["month"] == m) & (pnl["metric_name"] == metric_name)]
            result.append(
                _to_python(round(row["metric_value"].iloc[0], 2)) if not row.empty else 0
            )
        return result

    return {
        "months": [pd.Timestamp(m).strftime("%Y-%m") for m in months_sorted],
        "revenue": _series("revenue"),
        "net_profit": _series("net_profit"),
        "ebit": _series("ebit"),
    }


def anonymize_names_in_text(text: str, supabase: Client) -> str:
    """Replace real names in user input with anonymous IDs before sending to LLM.

    Sorts by name length descending to avoid partial replacements
    (e.g. "Маріяна" matched before "Марія").

    Args:
        text: Raw user input that may contain real names.
        supabase: Authenticated Supabase client.

    Returns:
        Text with real names replaced by anonymous IDs.
    """
    import re

    response = supabase.table("persons").select("anonymous_id, real_name").execute()
    pairs = [
        (row["real_name"], row["anonymous_id"])
        for row in (response.data or [])
        if row.get("real_name") and row.get("anonymous_id")
    ]
    pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for real_name, anon_id in pairs:
        text = re.sub(re.escape(real_name), anon_id, text, flags=re.IGNORECASE)
    return text


def resolve_names_in_text(text: str, supabase: Client) -> str:
    """Replace all anonymous IDs in text with real names.

    Called after LLM generates a response — the LLM never sees real names,
    substitution happens on the server before showing the result to the user.

    Args:
        text: Raw LLM output that may contain anonymous IDs like "rehab_01".
        supabase: Authenticated Supabase client.

    Returns:
        Text with anonymous IDs replaced by real names where a mapping exists.
    """
    if isinstance(text, list):
        text = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in text)
    name_map = _fetch_name_map(supabase)
    for anon_id, real_name in name_map.items():
        text = text.replace(anon_id, real_name)
    return text


def fetch_available_months(supabase: Client) -> list[str]:
    """Return distinct months present in monthly_metrics, newest first.

    Returns:
        List of month strings in YYYY-MM format, sorted descending.
    """
    metrics = _fetch_all_metrics(supabase)
    if metrics.empty:
        return []
    months = sorted(metrics["month"].dt.to_period("M").unique(), reverse=True)
    return [str(m) for m in months]


def build_dashboard_data(supabase: Client, target_month: str) -> dict:
    """Assemble all data required to render the financial dashboard.

    Args:
        supabase: Authenticated Supabase client.
        target_month: Month string in YYYY-MM format.

    Returns:
        Dictionary with keys: ``empty``, ``target_month``, ``kpi``,
        ``history``, ``efficiency``, ``services``, ``salaries``,
        ``expenses_by_category``.
    """
    target_ts = pd.Timestamp(target_month + "-01")
    prev_ts = target_ts - pd.DateOffset(months=1)

    all_metrics = _fetch_all_metrics(supabase)
    expense_df = _fetch_expense_rows(supabase)

    if all_metrics.empty:
        return {"empty": True, "target_month": target_month}

    cur = _metrics_for_month(all_metrics, target_ts)
    prev = _metrics_for_month(all_metrics, prev_ts)

    rehab_persons = set(
        all_metrics.loc[
            all_metrics["metric_name"] == "service_count", "person"
        ]
        .dropna()
        .unique()
    )

    cur_pnl = _scalar_dict(cur, "pnl")
    prev_pnl = _scalar_dict(prev, "pnl")
    cur_cf = _scalar_dict(cur, "cashflow")
    prev_cf = _scalar_dict(prev, "cashflow")

    def _delta(cur_d: dict, prev_d: dict, key: str) -> float | None:
        c, p = cur_d.get(key, 0), prev_d.get(key, 0)
        return round(c - p, 2) if p else None

    def _delta_pct(cur_d: dict, prev_d: dict, key: str) -> float | None:
        c, p = cur_d.get(key, 0), prev_d.get(key, 0)
        if not p:
            return None
        return round((c - p) / abs(p) * 100, 1)

    def _delta_pp(cur_d: dict, prev_d: dict, key: str) -> float | None:
        """Delta in percentage points — for ratio/margin metrics."""
        if key not in prev_d:
            return None
        return round(cur_d.get(key, 0) - prev_d.get(key, 0), 1) or None

    salary_data = _build_salaries(cur)

    kpi = {
        "revenue": cur_pnl.get("revenue", 0),
        "revenue_delta": _delta(cur_pnl, prev_pnl, "revenue"),
        "revenue_delta_pct": _delta_pct(cur_pnl, prev_pnl, "revenue"),
        "ebit": cur_pnl.get("ebit", 0),
        "ebit_delta": _delta(cur_pnl, prev_pnl, "ebit"),
        "ebit_delta_pct": _delta_pct(cur_pnl, prev_pnl, "ebit"),
        "operating_cf": cur_cf.get("operating_cf", 0),
        "operating_cf_delta": _delta(cur_cf, prev_cf, "operating_cf"),
        "operating_cf_delta_pct": _delta_pct(cur_cf, prev_cf, "operating_cf"),
        "operating_margin_pct": cur_pnl.get("operating_margin_pct", 0),
        "operating_margin_pct_delta": _delta_pp(cur_pnl, prev_pnl, "operating_margin_pct"),
        "expenses": cur_pnl.get("total_expenses", 0),
        "salary_total": salary_data["total"],
    }

    pnl_all = all_metrics[
        (all_metrics["category"] == "pnl") & all_metrics["person"].isna()
    ]
    cf_all = all_metrics[
        (all_metrics["category"] == "cashflow") & all_metrics["person"].isna()
    ]
    months_sorted = sorted(pnl_all["month"].unique())
    month_labels = [pd.Timestamp(m).strftime("%Y-%m") for m in months_sorted]

    def _series(metric_name: str) -> list[float]:
        result = []
        for m in months_sorted:
            row = pnl_all[
                (pnl_all["month"] == m) & (pnl_all["metric_name"] == metric_name)
            ]
            result.append(
                _to_python(round(row["metric_value"].iloc[0], 2)) if not row.empty else 0
            )
        return result

    def _cf_series(metric_name: str) -> list[float]:
        result = []
        for m in months_sorted:
            row = cf_all[
                (cf_all["month"] == m) & (cf_all["metric_name"] == metric_name)
            ]
            result.append(
                _to_python(round(row["metric_value"].iloc[0], 2)) if not row.empty else 0
            )
        return result

    sal_all = all_metrics[all_metrics["metric_name"] == "salary_amount"]
    salary_totals = [
        _to_python(
            round(sal_all[sal_all["month"] == m]["metric_value"].sum(), 2)
        )
        for m in months_sorted
    ]

    history = {
        "months": month_labels,
        "revenue": _series("revenue"),
        "ebit": _series("ebit"),
        "operating_cf": _cf_series("operating_cf"),
        "expenses": _series("total_expenses"),
        "salary_totals": salary_totals,
        "operating_margin": _series("operating_margin_pct"),
    }

    efficiency_raw = _build_efficiency(cur)
    efficiency = {k: v for k, v in efficiency_raw.items() if k in rehab_persons}

    services_raw = _build_services(cur)
    services_by_emp = {
        k: v
        for k, v in services_raw.get("by_employee", {}).items()
        if k in rehab_persons
    }
    services = {
        "total": sum(services_by_emp.values()) if services_by_emp else 0,
        "by_employee": services_by_emp,
    }

    name_map = _fetch_name_map(supabase)

    all_person_ids = set(efficiency.keys()) | set(
        services_raw.get("by_employee", {}).keys()
    ) | set(salary_data["top_3"].keys())
    names_missing = bool(all_person_ids and not name_map)

    return {
        "empty": False,
        "target_month": target_month,
        "kpi": kpi,
        "history": history,
        "efficiency": _apply_names(efficiency, name_map),
        "services": {
            "total": services["total"],
            "by_employee": _apply_names(services["by_employee"], name_map),
        },
        "salaries": {
            "total": salary_data["total"],
            "top_3": _apply_names(salary_data["top_3"], name_map),
        },
        "service_types": _build_service_types(cur),
        "expenses_by_category": _build_expense_detail(expense_df, target_ts),
        "daily_services": _fetch_daily_services(supabase, target_ts),
        "names_missing": names_missing,
    }


def build_agent_context(supabase: Client, target_month: str) -> dict:
    """Assemble the structured context dict used by report.py.

    Args:
        supabase: Authenticated Supabase client.
        target_month: Month string in YYYY-MM format.

    Returns:
        Dictionary with ``target_month``, ``current_month_summary``, and
        ``history_summary`` keys.
    """
    target_ts = pd.Timestamp(target_month + "-01")

    all_metrics = _fetch_all_metrics(supabase)
    expense_df = _fetch_expense_rows(supabase)

    if all_metrics.empty:
        return {
            "target_month": target_month,
            "current_month_summary": {},
            "history_summary": {"months": []},
        }

    mm = _metrics_for_month(all_metrics, target_ts)
    pnl_scalars = _scalar_dict(mm, "pnl")

    current = {
        "pnl": pnl_scalars,
        "cashflow": _scalar_dict(mm, "cashflow"),
        "salaries": _build_salaries(mm),
        "services": _build_services(mm),
        "efficiency": _build_efficiency(mm),
        "center_load": _build_center_load(mm),
        "expenses": _build_expense_detail(expense_df, target_ts),
        "amortization": {"total": pnl_scalars.get("amortization", 0)},
    }

    return {
        "target_month": target_month,
        "current_month_summary": current,
        "history_summary": _build_history(all_metrics, target_ts),
    }
