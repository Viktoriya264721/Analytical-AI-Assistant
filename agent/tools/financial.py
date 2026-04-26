from __future__ import annotations

import datetime
import json
from typing import Callable

from dateutil.relativedelta import relativedelta
from langchain_core.tools import tool
from supabase import Client

_PNL_METRICS = {
    "revenue", "card_revenue", "cash_revenue",
    "specialist_payouts_core", "support_salaries",
    "gross_profit", "gross_margin_pct",
    "expenses_amount", "total_expenses",
    "amortization",
    "ebit", "operating_margin_pct",
    "owner_share_33",
}

_CASHFLOW_METRICS = {
    "operating_inflow",
    "specialist_payouts_outflow", "support_salaries_outflow", "opex_outflow",
    "operating_cf", "capex",
}

_SIGNIFICANT_CHANGE_THRESHOLD = 0.15


def _month_to_date(month: str) -> str:
    """Convert ``YYYY-MM`` to the ``YYYY-MM-01`` format stored in Supabase."""
    return f"{month}-01"


def _prev_month(month: str) -> str:
    """Return the preceding month in ``YYYY-MM`` format."""
    date = datetime.date.fromisoformat(_month_to_date(month))
    return (date - relativedelta(months=1)).strftime("%Y-%m")


def _fetch_scalar_metrics(
    supabase: Client,
    month: str,
    category: str,
    expected: set[str],
) -> dict[str, float]:
    """Fetch aggregate (non-person) metrics for *month* and *category*.

    Args:
        supabase: Authenticated Supabase client.
        month: Month in ``YYYY-MM`` format.
        category: Metric category, e.g. ``"pnl"`` or ``"cashflow"``.
        expected: Set of metric names to extract.

    Returns:
        Mapping of metric name to float value; missing metrics default to 0.
    """
    response = (
        supabase.table("monthly_metrics")
        .select("metric_name, metric_value")
        .eq("month", _month_to_date(month))
        .eq("category", category)
        .is_("person", "null")
        .execute()
    )
    data = {row["metric_name"]: row["metric_value"] for row in (response.data or [])}
    return {k: float(data.get(k, 0)) for k in expected}


def _compute_delta(current: float, previous: float) -> dict[str, float]:
    """Return absolute and percentage delta between two values."""
    delta = round(current - previous, 2)
    delta_pct = round((delta / abs(previous)) * 100, 1) if previous else 0.0
    return {"delta": delta, "delta_pct": delta_pct}


def make_financial_tools(supabase: Client) -> list:
    """Create P&L and cash-flow tools with an injected Supabase client.

    Args:
        supabase: Authenticated Supabase client shared across all tools.

    Returns:
        List of LangChain tool callables ready to bind to an LLM.
    """

    @tool
    def get_pnl(month: str) -> str:
        """Get the full Profit & Loss report for a given month.

        Includes revenue breakdown (card / cash), salary split (rehab vs
        non-rehab), owner fees, gross/operating/net profit, margins, and a
        month-over-month comparison against the previous month.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with all P&L metrics and ``vs_prev_month`` deltas.
        """
        current = _fetch_scalar_metrics(supabase, month, "pnl", _PNL_METRICS)
        _PREV_COMPARE = {"revenue", "ebit", "total_expenses", "specialist_payouts_core"}
        previous = _fetch_scalar_metrics(supabase, _prev_month(month), "pnl", _PREV_COMPARE)

        vs_prev: dict[str, dict] = {}
        for key in _PREV_COMPARE:
            if previous.get(key):
                vs_prev[f"{key}_delta"] = _compute_delta(current[key], previous[key])

        result = {"month": month, **current, "vs_prev_month": vs_prev}
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_cashflow(month: str) -> str:
        """Get the cash-flow statement for a given month.

        Returns operating inflows, salary/owner/opex outflows, operating
        cash flow, capital expenditure, and net cash flow.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with all cash-flow metrics.
        """
        metrics = _fetch_scalar_metrics(supabase, month, "cashflow", _CASHFLOW_METRICS)
        return json.dumps({"month": month, **metrics}, ensure_ascii=False)

    @tool
    def get_daily_revenue(date: str) -> str:
        """Get revenue for a specific calendar day.

        Use this when asked about a concrete date, e.g. "скільки заробили
        2 червня?" or "карткою 6 червня скільки прийшло?".

        Args:
            date: Date in YYYY-MM-DD format, e.g. ``"2025-06-02"``.

        Returns:
            JSON string with ``total_revenue``, ``card_revenue``, and
            ``cash_revenue`` for that day, or ``{"found": false}`` when
            the date has no record.
        """
        response = (
            supabase.table("daily_revenue")
            .select("date, total_revenue, card_revenue, cash_revenue")
            .eq("date", date)
            .execute()
        )
        rows = response.data or []
        if not rows:
            return json.dumps({"date": date, "found": False}, ensure_ascii=False)
        row = rows[0]
        return json.dumps(
            {
                "date": date,
                "found": True,
                "total_revenue": float(row["total_revenue"] or 0),
                "card_revenue": float(row["card_revenue"] or 0),
                "cash_revenue": float(row["cash_revenue"] or 0),
            },
            ensure_ascii=False,
        )

    @tool
    def get_revenue_extremes() -> str:
        """Find the best and worst revenue days across the entire history.

        Use this when asked "який день найкращий / найгірший по виручці за весь час?"

        Returns:
            JSON string with ``best`` and ``worst`` entries, each containing
            ``date`` (YYYY-MM-DD) and ``total_revenue``.
        """
        response = (
            supabase.table("daily_revenue")
            .select("date, total_revenue")
            .execute()
        )
        rows = response.data or []
        if not rows:
            return json.dumps({"found": False}, ensure_ascii=False)
        best = max(rows, key=lambda r: float(r["total_revenue"]))
        worst = min(rows, key=lambda r: float(r["total_revenue"]))
        return json.dumps(
            {
                "best":  {"date": best["date"],  "total_revenue": float(best["total_revenue"])},
                "worst": {"date": worst["date"], "total_revenue": float(worst["total_revenue"])},
            },
            ensure_ascii=False,
        )

    return [get_pnl, get_cashflow, get_daily_revenue, get_revenue_extremes]
