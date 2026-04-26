from __future__ import annotations

import json
from collections import defaultdict

from langchain_core.tools import tool
from supabase import Client

from agent.tools.financial import _month_to_date


def _fetch_person_metrics(
    supabase: Client,
    month: str,
    metric_name: str,
) -> dict[str, float]:
    """Fetch per-person metric values for a given month.

    Args:
        supabase: Authenticated Supabase client.
        month: Month in ``YYYY-MM`` format.
        metric_name: The ``metric_name`` value to filter on.

    Returns:
        Mapping of anonymised person ID to float value.
    """
    response = (
        supabase.table("monthly_metrics")
        .select("person, metric_value")
        .eq("month", _month_to_date(month))
        .eq("metric_name", metric_name)
        .not_.is_("person", "null")
        .execute()
    )
    return {
        row["person"]: float(row["metric_value"])
        for row in (response.data or [])
        if row.get("person")
    }


def _fetch_scalar(
    supabase: Client,
    month: str,
    metric_name: str,
) -> float:
    """Fetch a single aggregate (non-person) metric value.

    Args:
        supabase: Authenticated Supabase client.
        month: Month in ``YYYY-MM`` format.
        metric_name: The ``metric_name`` value to look up.

    Returns:
        Float metric value, or ``0.0`` when the metric is absent.
    """
    response = (
        supabase.table("monthly_metrics")
        .select("metric_value")
        .eq("month", _month_to_date(month))
        .eq("metric_name", metric_name)
        .is_("person", "null")
        .execute()
    )
    rows = response.data or []
    return float(rows[0]["metric_value"]) if rows else 0.0


def make_people_tools(supabase: Client) -> list:
    """Create salary, efficiency, and services tools with an injected Supabase client.

    Args:
        supabase: Authenticated Supabase client shared across all tools.

    Returns:
        List of LangChain tool callables ready to bind to an LLM.
    """

    @tool
    def get_salaries(month: str) -> str:
        """Get the salary breakdown for a given month.

        Returns the total salary expense, per-employee amounts, and the
        top-3 highest-paid employees. Employee identifiers are anonymised.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``total``, ``by_employee``, and ``top_3`` keys.
        """
        by_person = _fetch_person_metrics(supabase, month, "salary_amount")

        total = round(sum(by_person.values()), 2)
        top_3 = dict(
            sorted(by_person.items(), key=lambda kv: kv[1], reverse=True)[:3]
        )

        result = {
            "month": month,
            "total": total,
            "by_employee": by_person,
            "top_3": top_3,
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_efficiency(month: str) -> str:
        """Get per-employee efficiency percentages for a given month.

        Efficiency is defined as the ratio of actual service hours to
        scheduled (available) hours, expressed as a percentage. Only
        rehab staff members have an efficiency metric.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``by_employee`` mapping person IDs to
            their efficiency percentage.
        """
        by_person = _fetch_person_metrics(supabase, month, "efficiency_pct")
        result = {"month": month, "by_employee": by_person}
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_services(month: str) -> str:
        """Get rehab service counts for a given month.

        Returns the total number of services delivered and a per-employee
        breakdown. Only rehab staff members appear in this report.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``total`` (aggregate) and ``by_employee``
            (per-person service counts).
        """
        total = _fetch_scalar(supabase, month, "total_services")
        by_person = _fetch_person_metrics(supabase, month, "service_count")

        result = {
            "month": month,
            "total": total,
            "by_employee": by_person,
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_revenue_by_person(month: str) -> str:
        """Get revenue brought by each rehab employee for a given month.

        Returns per-employee income amounts, sorted from highest to lowest.
        Only rehab staff have this metric — non-rehab employees will not appear.

        Args:
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``by_employee`` mapping person IDs to income amount,
            and ``top_earner`` with the highest-contributing employee.
        """
        by_person = _fetch_person_metrics(supabase, month, "person_income")

        if not by_person:
            return json.dumps({"month": month, "by_employee": {}, "top_earner": None},
                              ensure_ascii=False)

        sorted_persons = dict(sorted(by_person.items(), key=lambda kv: kv[1], reverse=True))
        top_earner = next(iter(sorted_persons))

        result = {
            "month":       month,
            "by_employee": sorted_persons,
            "top_earner":  {"person": top_earner, "income": sorted_persons[top_earner]},
        }
        return json.dumps(result, ensure_ascii=False)

    @tool
    def get_person_activity(person_id: str, month: str) -> str:
        """Get the breakdown of service units performed by a specific specialist for a given month.

        Use this when asked how many sessions / units / services of a specific
        type (massage, LFK, physio...) a named specialist delivered.
        Always call ``find_person`` first to obtain the ``person_id``.

        Args:
            person_id: The anonymous identifier of the specialist,
                e.g. ``"rehab_01"``.
            month: Target month in YYYY-MM format, e.g. ``"2025-11"``.

        Returns:
            JSON string with ``total_units`` (int) and ``by_type``
            (mapping service type → unit count).
        """
        year, mon = month.split("-")
        date_gte = f"{year}-{mon}-01"
        next_year = int(year) + (int(mon) // 12)
        next_mon = (int(mon) % 12) + 1
        date_lt = f"{next_year:04d}-{next_mon:02d}-01"

        response = (
            supabase.table("specialist_activity")
            .select("activity_type, units")
            .eq("person", person_id)
            .gte("date", date_gte)
            .lt("date", date_lt)
            .execute()
        )

        by_type: dict[str, int] = {}
        for row in response.data or []:
            atype = row.get("activity_type") or "unknown"
            by_type[atype] = by_type.get(atype, 0) + int(row.get("units") or 0)

        total = sum(by_type.values())
        return json.dumps(
            {"month": month, "person": person_id, "total_units": total, "by_type": by_type},
            ensure_ascii=False,
        )

    @tool
    def get_specialist_capacity(person_id: str, month: str) -> str:
        """Get the scheduled (available) hours for a specific specialist in a given month.

        Use this when asked how many hours a specialist was available / scheduled,
        or to compute their utilisation rate manually.
        Always call ``find_person`` first to obtain the ``person_id``.

        Args:
            person_id: The anonymous identifier of the specialist,
                e.g. ``"rehab_01"``.
            month: Target month in YYYY-MM format, e.g. ``"2025-10"``.

        Returns:
            JSON string with ``available_hours`` (float) for the month.
        """
        year, mon = month.split("-")
        date_gte = f"{year}-{mon}-01"
        next_year = int(year) + (int(mon) // 12)
        next_mon = (int(mon) % 12) + 1
        date_lt = f"{next_year:04d}-{next_mon:02d}-01"

        response = (
            supabase.table("specialist_capacity")
            .select("available_hours")
            .eq("person", person_id)
            .gte("date", date_gte)
            .lt("date", date_lt)
            .execute()
        )

        total_hours = sum(float(r["available_hours"]) for r in (response.data or []))
        return json.dumps(
            {"month": month, "person": person_id, "available_hours": total_hours},
            ensure_ascii=False,
        )

    @tool
    def get_person_summary(person_id: str, months: list[str]) -> str:
        """Get salary, revenue, and service count for a person across multiple months.

        Use this for questions that span several months: "як росла марія",
        "хто більше отримав за весь рік", "порівняй двох людей за весь 2025".
        One database call replaces N separate get_salaries / get_revenue_by_person calls.
        Always call ``find_person`` first to obtain the ``person_id``.

        Args:
            person_id: The anonymous identifier, e.g. ``"rehab_01"``.
            months: List of months in YYYY-MM format, e.g.
                ``["2025-06", "2025-07", "2025-08"]``.

        Returns:
            JSON string with a ``by_month`` dict mapping each month to
            ``{salary, revenue, service_count}``, plus ``totals`` with
            the sum of each metric across all requested months.
        """
        dates = [_month_to_date(m) for m in months]
        response = (
            supabase.table("monthly_metrics")
            .select("month, metric_name, metric_value")
            .eq("person", person_id)
            .in_("month", dates)
            .in_("metric_name", ["salary_amount", "person_income", "service_count"])
            .execute()
        )

        date_to_month = {_month_to_date(m): m for m in months}
        by_month: dict[str, dict[str, float]] = {m: {"salary": 0.0, "revenue": 0.0, "service_count": 0.0} for m in months}

        for row in response.data or []:
            month = date_to_month.get(row["month"])
            if not month:
                continue
            val = float(row["metric_value"] or 0)
            if row["metric_name"] == "salary_amount":
                by_month[month]["salary"] = val
            elif row["metric_name"] == "person_income":
                by_month[month]["revenue"] = val
            elif row["metric_name"] == "service_count":
                by_month[month]["service_count"] = val

        totals = {
            "salary": round(sum(v["salary"] for v in by_month.values()), 2),
            "revenue": round(sum(v["revenue"] for v in by_month.values()), 2),
            "service_count": round(sum(v["service_count"] for v in by_month.values()), 2),
        }

        return json.dumps(
            {"person": person_id, "months": months, "by_month": by_month, "totals": totals},
            ensure_ascii=False,
        )

    @tool
    def get_person_activity_trend(person_id: str, months: list[str]) -> str:
        """Get monthly activity breakdown for a person across multiple months.

        Use this when asked about a person's unit dynamics over time:
        "як росла марія по одиницях", "динаміка послуг за рік",
        "скільки одиниць робив дмитро щомісяця".
        One database call covers all requested months.
        Always call ``find_person`` first to obtain the ``person_id``.

        Args:
            person_id: The anonymous identifier, e.g. ``"rehab_01"``.
            months: List of months in YYYY-MM format, e.g.
                ``["2025-06", "2025-07", "2025-08"]``.

        Returns:
            JSON string with ``by_month`` mapping each month to
            ``{by_type: {service_type: count}, total_units: int}``.
        """
        months_sorted = sorted(months)
        date_gte = f"{months_sorted[0]}-01"
        last = months_sorted[-1]
        year, mon = last.split("-")
        next_year = int(year) + (int(mon) // 12)
        next_mon = (int(mon) % 12) + 1
        date_lt = f"{next_year:04d}-{next_mon:02d}-01"

        response = (
            supabase.table("specialist_activity")
            .select("date, activity_type, units")
            .eq("person", person_id)
            .gte("date", date_gte)
            .lt("date", date_lt)
            .execute()
        )

        months_set = set(months)
        by_month: dict[str, dict] = defaultdict(lambda: defaultdict(int))
        for row in response.data or []:
            month = row["date"][:7]
            if month in months_set:
                by_month[month][row.get("activity_type") or "unknown"] += int(row.get("units") or 0)

        result_by_month = {}
        for m in months:
            types = dict(by_month.get(m, {}))
            result_by_month[m] = {"by_type": types, "total_units": sum(types.values())}

        return json.dumps(
            {"person": person_id, "months": months, "by_month": result_by_month},
            ensure_ascii=False,
        )

    return [get_salaries, get_efficiency, get_services, get_revenue_by_person,
            get_person_activity, get_specialist_capacity, get_person_summary,
            get_person_activity_trend]
