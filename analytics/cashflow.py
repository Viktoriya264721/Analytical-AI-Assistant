from typing import List, Dict
import pandas as pd

from analytics.queries import filter_month


def _lookup(pnl_metrics: List[Dict], name: str) -> float:
    """Finds a metric value by name in P&L results."""
    for m in pnl_metrics:
        if m["metric_name"] == name:
            return m["metric_value"] or 0.0
    return 0.0


def compute_cashflow(
    tables: Dict[str, pd.DataFrame],
    month: pd.Timestamp,
    pnl_metrics: List[Dict],
) -> List[Dict]:
    """Compute Cash Flow metrics for one month using pre-computed P&L values."""
    revenue = _lookup(pnl_metrics, "revenue")
    specialist_payouts_core = _lookup(pnl_metrics, "specialist_payouts_core")
    support_salaries = _lookup(pnl_metrics, "support_salaries")
    expenses_amount = _lookup(pnl_metrics, "expenses_amount")

    operating_inflow = revenue
    specialist_payouts_outflow = specialist_payouts_core
    support_salaries_outflow = support_salaries
    opex_outflow = expenses_amount
    operating_cf = round(
        operating_inflow
        - specialist_payouts_outflow
        - support_salaries_outflow
        - opex_outflow,
        2,
    )

    amortization_m = filter_month(tables.get("amortization", pd.DataFrame()), month)
    capex = round(float(amortization_m["total_amount"].sum()), 2) if not amortization_m.empty else 0.0

    def m(name, value):
        return {
            "month": month,
            "metric_name": name,
            "metric_value": value,
            "person": None,
            "category": "cashflow",
        }

    return [
        m("operating_inflow", operating_inflow),
        m("specialist_payouts_outflow", specialist_payouts_outflow),
        m("support_salaries_outflow", support_salaries_outflow),
        m("opex_outflow", opex_outflow),
        m("operating_cf", operating_cf),
        m("capex", capex),
    ]
