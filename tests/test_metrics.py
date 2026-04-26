import csv
import os
import sys
from datetime import datetime
from pathlib import Path

env_path = os.getenv("ENV_PATH")
if env_path:
    from dotenv import load_dotenv
    load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase import create_client

EXPECTED_FILE = Path(__file__).parent / "expected_metrics.csv"
RESULTS_DIR   = Path(__file__).parent / "results"
FLOAT_TOL     = 0.05 

def load_expected() -> list[dict]:
    with open(EXPECTED_FILE, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_persons(supabase) -> dict:
    """Return ({anonymous_id: real_name}, {real_name_lower: anonymous_id}) mappings from the persons table."""
    resp = supabase.table("persons").select("real_name, anonymous_id").execute()
    id_to_name = {}
    name_to_id = {}
    for row in (resp.data or []):
        anon = row["anonymous_id"]
        name = row["real_name"].strip().lower()
        id_to_name[anon] = name
        name_to_id[name] = anon
    return id_to_name, name_to_id


def load_monthly_metrics(supabase) -> list[dict]:
    """Load all rows from monthly_metrics using pagination."""
    rows, offset, page = [], 0, 500
    while True:
        resp = (
            supabase.table("monthly_metrics")
            .select("month, metric_name, metric_value, person, category")
            .range(offset, offset + page - 1)
            .execute()
        )
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < page:
            break
        offset += page
    return rows


def index_db_metrics(db_rows: list[dict], id_to_name: dict) -> dict:
    """Index DB metrics into a dict keyed by (month_str, metric_name, person_real_name_lower, category)."""
    index = {}
    for row in db_rows:
        month = str(row["month"])[:7]
        metric = row["metric_name"]
        anon   = row.get("person") or ""
        person = id_to_name.get(anon, "").lower() if anon else ""
        cat    = (row.get("category") or "").strip().lower()
        value  = row.get("metric_value")
        key = (month, metric, person, cat)
        index[key] = value
    return index



def compare(expected: list[dict], db_index: dict, name_to_id: dict) -> list[dict]:
    results = []

    for row in expected:
        month  = row["month"]
        metric = row["metric_name"]
        person = row["person"].strip().lower()
        cat    = row["category"].strip().lower()
        exp_v  = float(row["expected_value"])

        key = (month, metric, person, cat)
        db_v = db_index.get(key)

        if db_v is None:
            status = "MISSING"
            got    = None
            diff   = None
        else:
            db_v  = float(db_v)
            diff  = abs(exp_v - db_v)
            status = "OK" if diff <= FLOAT_TOL else "MISMATCH"
            got   = db_v

        results.append({
            "month":    month,
            "metric":   metric,
            "person":   person,
            "category": cat,
            "expected": exp_v,
            "got":      got,
            "diff":     diff,
            "status":   status,
        })

    return results



def save_detail_csv(results: list[dict], run_ts: datetime) -> Path:
    """Write a detail CSV with one row per comparison result."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"metrics_detail_{run_ts.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["month", "metric", "person", "category",
                                                "expected", "got", "diff", "status"])
        writer.writeheader()
        writer.writerows(results)
    return out


def save_summary_csv(results: list[dict], run_ts: datetime) -> Path:
    """Write a summary CSV aggregating check counts by metric."""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"metrics_summary_{run_ts.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    from collections import defaultdict
    by_metric: dict[str, list] = defaultdict(list)
    for r in results:
        by_metric[r["metric"]].append(r)

    ts = run_ts.strftime("%Y-%m-%d %H:%M:%S")
    summary_rows = []
    for metric, items in sorted(by_metric.items()):
        total    = len(items)
        ok       = sum(1 for i in items if i["status"] == "OK")
        missing  = sum(1 for i in items if i["status"] == "MISSING")
        mismatch = sum(1 for i in items if i["status"] == "MISMATCH")
        pct      = round(ok / total * 100, 1) if total else 0.0
        passed   = "TRUE" if mismatch == 0 and missing == 0 else "FALSE"
        summary_rows.append({
            "timestamp":       ts,
            "metric":          metric,
            "total_checks":    total,
            "ok":              ok,
            "missing":         missing,
            "mismatch":        mismatch,
            "accuracy_pct":    pct,
            "passed":          passed,
        })

    total_all    = len(results)
    ok_all       = sum(1 for r in results if r["status"] == "OK")
    missing_all  = sum(1 for r in results if r["status"] == "MISSING")
    mismatch_all = sum(1 for r in results if r["status"] == "MISMATCH")
    summary_rows.append({
        "timestamp":    ts,
        "metric":       "ВСЬОГО",
        "total_checks": total_all,
        "ok":           ok_all,
        "missing":      missing_all,
        "mismatch":     mismatch_all,
        "accuracy_pct": round(ok_all / total_all * 100, 1) if total_all else 0.0,
        "passed":       "TRUE" if mismatch_all == 0 and missing_all == 0 else "FALSE",
    })

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "metric", "total_checks",
                                                "ok", "missing", "mismatch",
                                                "accuracy_pct", "passed"])
        writer.writeheader()
        writer.writerows(summary_rows)
    return out


def print_report(results: list[dict]):
    total    = len(results)
    ok       = sum(1 for r in results if r["status"] == "OK")
    missing  = sum(1 for r in results if r["status"] == "MISSING")
    mismatch = sum(1 for r in results if r["status"] == "MISMATCH")
    pct      = ok / total * 100 if total else 0

    print()
    print("=" * 70)
    print("  РЕЗУЛЬТАТИ ПЕРЕВІРКИ МЕТРИК monthly_metrics")
    print("=" * 70)
    print(f"  Всього перевірок : {total}")
    print(f"  Коректних (OK)   : {ok}  ({pct:.1f}%)")
    print(f"  Відсутніх        : {missing}")
    print(f"  Розбіжностей     : {mismatch}")
    print(f"  Загальний статус : {'PASSED' if mismatch == 0 and missing == 0 else 'FAILED'}")
    print()

    if missing > 0:
        print("  ВІДСУТНІ в monthly_metrics:")
        for r in results:
            if r["status"] == "MISSING":
                p = f" | {r['person']}" if r["person"] else ""
                print(f"    {r['month']} | {r['metric']} | cat={r['category']}{p}")
        print()

    if mismatch > 0:
        print("  РОЗБІЖНОСТІ:")
        print(f"  {'Місяць':<9} {'Метрика':<28} {'Особа':<12} {'Очікувано':>12} {'Є в БД':>12} {'Δ':>10}")
        print("  " + "-" * 85)
        for r in results:
            if r["status"] == "MISMATCH":
                p = r["person"][:11] if r["person"] else ""
                print(
                    f"  {r['month']:<9} {r['metric']:<28} {p:<12} "
                    f"{r['expected']:>12.2f} {r['got']:>12.2f} {r['diff']:>10.2f}"
                )
    print("=" * 70)



def main():
    run_ts = datetime.now()
    print("=== Тест метрик: expected_metrics.csv vs Supabase monthly_metrics ===\n")

    if not EXPECTED_FILE.exists():
        print(f"ПОМИЛКА: {EXPECTED_FILE} не знайдено.")
        print("Спочатку запустіть: python tests/generate_expected_metrics.py")
        sys.exit(1)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ПОМИЛКА: SUPABASE_URL або SUPABASE_SERVICE_ROLE_KEY не задані.")
        sys.exit(1)

    supabase = create_client(url, key)
    print("Supabase: підключено")

    print("Завантаження expected_metrics.csv...", end=" ", flush=True)
    expected = load_expected()
    print(f"{len(expected)} рядків")

    print("Завантаження persons з Supabase...", end=" ", flush=True)
    id_to_name, name_to_id = load_persons(supabase)
    print(f"{len(id_to_name)} осіб")

    print("Завантаження monthly_metrics з Supabase...", end=" ", flush=True)
    db_rows = load_monthly_metrics(supabase)
    print(f"{len(db_rows)} рядків")

    print("Порівняння...", end=" ", flush=True)
    db_index = index_db_metrics(db_rows, id_to_name)
    results  = compare(expected, db_index, name_to_id)
    print("готово")

    expected_keys = {
        (r["month"], r["metric_name"], r["person"].strip().lower(), r["category"].strip().lower())
        for r in expected
    }
    extra = []
    for row in db_rows:
        month  = str(row["month"])[:7]
        metric = row["metric_name"]
        anon   = row.get("person") or ""
        person = id_to_name.get(anon, "").lower() if anon else ""
        cat    = (row.get("category") or "").strip().lower()
        key    = (month, metric, person, cat)
        if key not in expected_keys:
            extra.append(key)

    if extra:
        print(f"\n  УВАГА: {len(extra)} зайвих рядків у monthly_metrics (яких нема в expected):")
        for month, metric, person, cat in extra[:20]:
            p = f" | {person}" if person else ""
            print(f"    {month} | {metric} | cat={cat}{p}")
        if len(extra) > 20:
            print(f"    ... (+{len(extra)-20} ще)")

    print_report(results)

    detail_path  = save_detail_csv(results, run_ts)
    summary_path = save_summary_csv(results, run_ts)
    print(f"\n  Зведена таблиця : {summary_path}")
    print(f"  Деталі          : {detail_path}")


if __name__ == "__main__":
    main()
