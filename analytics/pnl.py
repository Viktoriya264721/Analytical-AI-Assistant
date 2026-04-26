from typing import List, Dict
import pandas as pd
from dateutil.relativedelta import relativedelta

from config.constants import OWNER_PROFIT_SHARE
from analytics.queries import filter_month, get_rehab_persons


def _compute_amortization(
    amortization_df: pd.DataFrame,
    target_month: pd.Timestamp,
) -> float:
    """
    Monthly amortization for target_month.

    Each asset contributes: total_amount / duration_months
    for each month in [purchase_date .. purchase_date + duration_months - 1].
    """
    if amortization_df.empty:
        return 0.0

    total = 0.0

    for _, row in amortization_df.iterrows():
        total_amount = row.get("total_amount")
        duration = row.get("duration_months")
        start_date = row.get("date")

        if pd.isna(total_amount) or pd.isna(duration) or pd.isna(start_date) or duration <= 0:
            continue

        start_month = start_date.to_period("M").to_timestamp()
        end_month = start_month + relativedelta(months=int(duration) - 1)

        if start_month <= target_month <= end_month:
            total += total_amount / duration

    return round(total, 2)


def compute_pnl(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
) -> List[Dict]:
    """
    Computes all P&L metrics for one month.

    Returns list of metric dicts ready for monthly_metrics upsert.
    """
    rehab_persons = get_rehab_persons(tables)

    revenue_m = filter_month(tables.get("daily_revenue", pd.DataFrame()), month)

    revenue = 0.0
    card_revenue = 0.0
    cash_revenue = 0.0

    if not revenue_m.empty:
        revenue = round(float(revenue_m["total_revenue"].fillna(0).sum()), 2)
        if "card_revenue" in revenue_m.columns:
            card_revenue = round(float(revenue_m["card_revenue"].fillna(0).sum()), 2)
        if "cash_revenue" in revenue_m.columns:
            cash_revenue = round(float(revenue_m["cash_revenue"].fillna(0).sum()), 2)

    payouts_m = filter_month(tables.get("specialist_payouts", pd.DataFrame()), month)

    specialist_payouts_core = 0.0
    support_salaries = 0.0

    if not payouts_m.empty and "payout_amount" in payouts_m.columns:
        rehab_mask = payouts_m["person"].isin(rehab_persons)
        specialist_payouts_core = round(
            float(payouts_m.loc[rehab_mask, "payout_amount"].fillna(0).sum()), 2
        )
        support_salaries = round(
            float(payouts_m.loc[~rehab_mask, "payout_amount"].fillna(0).sum()), 2
        )

    gross_profit = round(revenue - specialist_payouts_core, 2)
    gross_margin_pct = round((gross_profit / revenue) * 100, 1) if revenue else 0.0

    expenses_m = filter_month(tables.get("expenses", pd.DataFrame()), month)
    expenses_amount = round(float(expenses_m["amount"].fillna(0).sum()), 2) if not expenses_m.empty else 0.0
    total_expenses = round(expenses_amount + support_salaries, 2)

    amortization = _compute_amortization(
        tables.get("amortization", pd.DataFrame()), month
    )

    ebit = round(gross_profit - total_expenses - amortization, 2)
    operating_margin_pct = round((ebit / revenue) * 100, 1) if revenue else 0.0

    owner_share_33 = round(ebit * OWNER_PROFIT_SHARE, 2)

    def m(name, value):
        return {
            "month": month,
            "metric_name": name,
            "metric_value": value,
            "person": None,
            "category": "pnl",
        }

    return [
        m("revenue", revenue),
        m("card_revenue", card_revenue),
        m("cash_revenue", cash_revenue),
        m("specialist_payouts_core", specialist_payouts_core),
        m("support_salaries", support_salaries),
        m("gross_profit", gross_profit),
        m("gross_margin_pct", gross_margin_pct),
        m("expenses_amount", expenses_amount),
        m("total_expenses", total_expenses),
        m("amortization", amortization),
        m("ebit", ebit),
        m("operating_margin_pct", operating_margin_pct),
        m("owner_share_33", owner_share_33),
    ]
