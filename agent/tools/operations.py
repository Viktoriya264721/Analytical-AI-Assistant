from __future__ import annotations

import json

from langchain_core.tools import tool
from supabase import Client

from agent.tools.financial import _month_to_date


def _fetch_expense_categories(supabase: Client, month: str) -> dict[str, float]:
    """Fetch expense totals grouped by category for a given month.

    Queries the ``expenses`` domain table directly (not ``monthly_metrics``)
    to provide category-level detail not stored in aggregated metrics.

    Args:
        supabase: Authenticated Supabase client.
        month: Month in ``YYYY-MM`` format, e.g. ``"2025-11"``.

    Returns:
        Mapping of expense category name to total amount for the month.
    """
    year, mon = month.split("-")
    date_gte = f"{year}-{mon}-01"
    next_month_year = int(year) + (int(mon) // 12)
    next_month_mon = (int(mon) % 12) + 1
    date_lt = f"{next_month_year:04d}-{next_month_mon:02d}-01"

    response = (
        supabase.table("expenses")
        .select("amount, category")
        .gte("date", date_gte)
        .lt("date", date_lt)
        .execute()
    )

    totals: dict[str, float] = {}
    for row in response.data or []:
        cat = row.get("category") or "other"
        totals[cat] = round(totals.get(cat, 0.0) + float(row["amount"]), 2)
    return totals


def make_operations_tools(supabase: Client) -> list:
    """Create expense and service breakdown tools with an injected Supabase client.

    Args:
        supabase: Authenticated Supabase client shared across all tools.

    Returns:
        List of LangChain tool callables ready to bind to an LLM.
    """

    @tool
    def get_expenses(month: str) -> str:
        """Get the operating expense breakdown for a given month.

        Returns the total expense amount and a per-category breakdown,
        sorted from largest to smallest category spend.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``total`` and ``by_category`` (sorted descending
            by amount).
        """
        by_category = _fetch_expense_categories(supabase, month)
        total = round(sum(by_category.values()), 2)
        sorted_cats = dict(
            sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)
        )

        result = {
            "month": month,
            "total": total,
            "by_category": sorted_cats,
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_service_breakdown(month: str) -> str:
        """Get the breakdown of rehab services by type for a given month.

        Returns total units per service type, sorted from most to least performed.
        Use this to answer which services are most popular or most delivered.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``total`` units and ``by_service`` mapping
            service names to unit counts (sorted descending).
        """
        response = (
            supabase.table("monthly_metrics")
            .select("category, metric_value")
            .eq("month", _month_to_date(month))
            .eq("metric_name", "service_units")
            .is_("person", "null")
            .execute()
        )

        by_service = {
            row["category"]: float(row["metric_value"])
            for row in (response.data or [])
            if row.get("category")
        }

        sorted_services = dict(sorted(by_service.items(), key=lambda kv: kv[1], reverse=True))
        total = round(sum(sorted_services.values()), 2)

        result = {
            "month":      month,
            "total":      total,
            "by_service": sorted_services,
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_amortization(month: str) -> str:
        """Get asset amortization details for a given month.

        Returns the monthly amortization amount for each active asset and
        the total amortization for the month. An asset is active in a month
        if it was purchased on or before that month and has not fully
        amortized yet (purchase_date + duration_months > month).

        Use this when asked about depreciation of specific assets (e.g.
        "масажні столи на скільки амортизуються?") or total amortization
        for a month, or total asset purchase value.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-06"``.

        Returns:
            JSON string with:
            - ``total_amortization``: sum of monthly amounts for the month
            - ``total_assets_value``: sum of all asset purchase prices
            - ``by_asset``: list of active assets with ``name``,
              ``monthly_amount``, ``total_amount``, ``duration_months``,
              ``purchase_date``, ``months_remaining``
        """
        import datetime as _dt
        year, mon = month.split("-")
        month_end = _dt.date(int(year), int(mon), 1)

        response = (
            supabase.table("amortization")
            .select("asset_name, total_amount, duration_months, date")
            .execute()
        )

        by_asset = []
        total_amortization = 0.0
        total_assets_value = 0.0

        for row in response.data or []:
            purchase_date = _dt.date.fromisoformat(str(row["date"]))
            total_amount = float(row["total_amount"])
            duration = int(row["duration_months"])
            monthly = round(total_amount / duration, 2)
            months_elapsed = (month_end.year - purchase_date.year) * 12 + (
                month_end.month - purchase_date.month
            )

            total_assets_value += total_amount

            if months_elapsed < 0:
                continue
            months_remaining = duration - months_elapsed
            if months_remaining <= 0:
                continue

            total_amortization += monthly
            by_asset.append(
                {
                    "name": row["asset_name"],
                    "monthly_amount": monthly,
                    "total_amount": total_amount,
                    "duration_months": duration,
                    "purchase_date": str(purchase_date),
                    "months_remaining": months_remaining,
                }
            )

        by_asset.sort(key=lambda x: x["monthly_amount"], reverse=True)

        return json.dumps(
            {
                "month": month,
                "total_amortization": round(total_amortization, 2),
                "total_assets_value": round(total_assets_value, 2),
                "by_asset": by_asset,
            },
            ensure_ascii=False,
        )

    return [get_expenses, get_service_breakdown, get_amortization]
