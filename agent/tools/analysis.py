from __future__ import annotations

import json

from langchain_core.tools import tool
from supabase import Client

from agent.tools.financial import (
    _month_to_date,
    _prev_month,
    _fetch_scalar_metrics,
    _compute_delta,
    _PNL_METRICS,
    _SIGNIFICANT_CHANGE_THRESHOLD,
)

_TREND_METRICS = {"revenue", "ebit", "total_expenses", "specialist_payouts_core", "gross_profit"}


def _fetch_months_range(supabase: Client, months: list[str]) -> list[dict]:
    """Fetch P&L scalar metrics for a list of months.

    Args:
        supabase: Authenticated Supabase client.
        months: List of month strings in ``YYYY-MM`` format.

    Returns:
        List of dicts, each containing ``month`` and all ``_PNL_METRICS`` values.
    """
    dates = [_month_to_date(m) for m in months]
    response = (
        supabase.table("monthly_metrics")
        .select("month, metric_name, metric_value")
        .in_("month", dates)
        .eq("category", "pnl")
        .is_("person", "null")
        .execute()
    )

    from collections import defaultdict

    by_month: dict[str, dict[str, float]] = defaultdict(dict)
    for row in response.data or []:
        by_month[row["month"]][row["metric_name"]] = float(row["metric_value"])

    result = []
    for m, date in zip(months, dates):
        data = by_month.get(date, {})
        result.append({"month": m, **{k: data.get(k, 0.0) for k in _PNL_METRICS}})
    return result


def make_analysis_tools(supabase: Client) -> list:
    """Create analytical comparison and trend tools with an injected Supabase client.

    Args:
        supabase: Authenticated Supabase client shared across all tools.

    Returns:
        List of LangChain tool callables ready to bind to an LLM.
    """

    @tool
    def compare_months(month_a: str, month_b: str) -> str:
        """Compare P&L metrics between two months side-by-side.

        For each metric returns the value in both months and the absolute
        and percentage delta (month_b relative to month_a).

        Args:
            month_a: Reference month in YYYY-MM format, e.g. ``"2025-10"``.
            month_b: Comparison month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with a ``comparison`` dict mapping each metric name
            to ``{month_a, month_b, delta, delta_pct}``.
        """
        data_a = _fetch_scalar_metrics(supabase, month_a, "pnl", _PNL_METRICS)
        data_b = _fetch_scalar_metrics(supabase, month_b, "pnl", _PNL_METRICS)

        comparison: dict[str, dict] = {}
        for key in _PNL_METRICS:
            a, b = data_a[key], data_b[key]
            comparison[key] = {
                month_a: a,
                month_b: b,
                **_compute_delta(b, a),
            }

        result = {"month_a": month_a, "month_b": month_b, "comparison": comparison}
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_trend(metric: str, num_months: int, end_month: str) -> str:
        """Get the time-series trend for a single metric over recent months.

        Args:
            metric: Metric name, e.g. ``"revenue"``, ``"ebit"``,
                ``"total_expenses"``, ``"specialist_payouts_core"``,
                or ``"gross_profit"``.
            num_months: Number of months to include (most recent first),
                e.g. ``6`` for a half-year view.
            end_month: Last month of the range in YYYY-MM format,
                e.g. ``"2025-11"``.

        Returns:
            JSON string with ``metric``, ``months`` (list of YYYY-MM strings),
            and ``values`` (corresponding float list).
        """
        if metric not in _TREND_METRICS:
            return json.dumps(
                {
                    "error": f"Unknown metric '{metric}'. "
                    f"Supported: {sorted(_TREND_METRICS)}"
                }
            )

        months: list[str] = []
        current = end_month
        for _ in range(num_months):
            months.append(current)
            current = _prev_month(current)
        months.reverse()

        rows = _fetch_months_range(supabase, months)
        values = [round(r.get(metric, 0.0), 2) for r in rows]

        result = {"metric": metric, "months": months, "values": values}
        return json.dumps(result, ensure_ascii=False)

    @tool
    def detect_anomalies(month: str) -> str:
        """Detect significant month-over-month changes in key P&L metrics.

        A metric is flagged as an anomaly when its absolute percentage change
        vs the previous month exceeds 15 %.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with an ``anomalies`` list. Each entry contains
            ``metric``, ``current``, ``previous``, ``delta``, ``delta_pct``,
            and ``direction`` (``"increase"`` or ``"decrease"``).
        """
        current = _fetch_scalar_metrics(supabase, month, "pnl", _PNL_METRICS)
        previous = _fetch_scalar_metrics(
            supabase, _prev_month(month), "pnl", _PNL_METRICS
        )

        anomalies = []
        for key in _PNL_METRICS:
            cur_val, prev_val = current[key], previous[key]
            if not prev_val:
                continue
            info = _compute_delta(cur_val, prev_val)
            if abs(info["delta_pct"]) >= _SIGNIFICANT_CHANGE_THRESHOLD * 100:
                anomalies.append(
                    {
                        "metric": key,
                        "current": cur_val,
                        "previous": prev_val,
                        "delta": info["delta"],
                        "delta_pct": info["delta_pct"],
                        "direction": "increase" if info["delta"] > 0 else "decrease",
                    }
                )

        anomalies.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)
        result = {"month": month, "anomalies": anomalies}
        return json.dumps(result, ensure_ascii=False)

    return [compare_months, get_trend, detect_anomalies]
