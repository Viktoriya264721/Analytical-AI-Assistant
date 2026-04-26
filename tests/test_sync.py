import csv
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

env_path = os.getenv("ENV_PATH")
if env_path:
    from dotenv import load_dotenv
    load_dotenv(env_path)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from supabase import create_client

from ingestion.google_sheets import load_raw_tables
from processing.cleaning import clean_tables
from processing.normalization import normalize_tables
from processing.validation import validate_all
from processing.anonymization import anonymize_tables
from db.attach_identity import attach_identity
from storage.db_writer import upsert_all
from analytics.compute import compute_for_updated_months


TYPE_CHECKS = {
    "daily_revenue": {
        "date": "datetime64",
        "total_revenue": "float64",
    },
    "expenses": {
        "date": "datetime64",
        "amount": "float64",
    },
    "amortization": {
        "date": "datetime64",
        "total_amount": "float64",
        "duration_months": ("int", "Int"),
    },
    "specialist_capacity": {
        "date": "datetime64",
        "available_hours": "float64",
    },
    "specialist_activity": {
        "date": "datetime64",
        "units": ("int", "Int"),
    },
    "specialist_payouts": {
        "date": "datetime64",
        "payout_amount": "float64",
    },
}

ANON_PATTERN = re.compile(r"^(rehab|empl|owner)_\d+$")
SYNC_TIME_TARGET = 15.0

def check_structural_integrity(normalized_tables: dict) -> dict:
    df = normalized_tables.get("daily_revenue", pd.DataFrame())
    if df.empty:
        return {"score": None, "detail": "daily_revenue порожня"}

    has_both = df["card_revenue"].notna() & df["cash_revenue"].notna()
    sub = df[has_both].copy()

    if sub.empty:
        return {"score": 100.0, "detail": "немає рядків з card+cash (перевірка не потрібна)"}

    broken = sub[
        (sub["total_revenue"] - sub["card_revenue"] - sub["cash_revenue"]).abs() > 0.01
    ]
    score = (1 - len(broken) / len(sub)) * 100
    detail = f"{len(broken)} порушень з {len(sub)} рядків де є card+cash"
    return {"score": round(score, 1), "detail": detail}


def _dtype_matches(series: pd.Series, expected) -> bool:
    dtype_str = str(series.dtype).lower()
    if isinstance(expected, tuple):
        return any(e in dtype_str for e in expected)
    return expected in dtype_str


def check_data_types(normalized_tables: dict) -> dict:
    total, correct = 0, 0
    errors = []

    for table_name, col_checks in TYPE_CHECKS.items():
        df = normalized_tables.get(table_name, pd.DataFrame())
        for col, expected in col_checks.items():
            total += 1
            if df.empty or col not in df.columns:
                errors.append(f"{table_name}.{col}: колонка відсутня")
                continue
            if _dtype_matches(df[col], expected):
                correct += 1
            else:
                actual = str(df[col].dtype)
                errors.append(f"{table_name}.{col}: очікувався {expected}, є {actual}")

    score = (correct / total * 100) if total else 0.0
    return {
        "score": round(score, 1),
        "detail": "; ".join(errors) if errors else "всі типи коректні",
    }


def check_anonymization(anonymized_tables: dict) -> dict:
    total, anon = 0, 0
    errors = []

    for table_name, df in anonymized_tables.items():
        if df.empty or "person" not in df.columns:
            continue
        col = df["person"].dropna()
        for val in col:
            total += 1
            if ANON_PATTERN.match(str(val)):
                anon += 1
            else:
                errors.append(f"{table_name}: '{val}'")

    if total == 0:
        return {"score": None, "detail": "немає колонок person"}

    score = (anon / total) * 100
    detail = f"{anon}/{total} значень анонімізовано"
    if errors:
        detail += f"; не анонімізовано: {', '.join(errors[:5])}"
    return {"score": round(score, 1), "detail": detail}


def check_irreversibility(anonymized_tables: dict, supabase) -> dict:
    resp = supabase.table("persons").select("real_name").execute()
    real_names = {row["real_name"] for row in (resp.data or [])}

    if not real_names:
        return {"score": True, "detail": "таблиця persons порожня — нічого перевіряти"}

    leaks = []
    for table_name, df in anonymized_tables.items():
        for col in df.select_dtypes("object").columns:
            leaked = df[col].isin(real_names)
            if leaked.any():
                leaks.append(f"{table_name}.{col}: {leaked.sum()} витоків")

    ok = len(leaks) == 0
    detail = "реальних імен у domain-таблицях не знайдено" if ok else "; ".join(leaks)
    return {"score": ok, "detail": detail}


DB_TABLES = [
    "daily_revenue", "expenses", "amortization",
    "specialist_capacity", "specialist_activity", "specialist_payouts",
]


def check_record_loss(input_counts: dict, supabase) -> dict:
    results = {}
    total_in = total_db = 0

    for table in DB_TABLES:
        in_count = input_counts.get(table, 0)
        resp = supabase.table(table).select("id", count="exact").execute()
        db_count = resp.count if resp.count is not None else len(resp.data or [])

        results[table] = {"input": in_count, "db": db_count}
        total_in += in_count
        total_db += db_count

    diff = abs(total_in - total_db)
    score = (1 - diff / total_in) * 100 if total_in else 100.0

    rows = [f"{t}: {v['input']} → {v['db']}" for t, v in results.items()]
    detail = " | ".join(rows)
    return {"score": round(score, 1), "detail": detail}


_NUMERIC_FIELDS = {
    "daily_revenue":       ["total_revenue", "card_revenue", "cash_revenue"],
    "expenses":            ["amount"],
    "amortization":        ["total_amount", "duration_months"],
    "specialist_capacity": ["available_hours"],
    "specialist_activity": ["units"],
    "specialist_payouts":  ["payout_amount", "generated_revenue"],
}


_TEXT_FIELDS = {
    "expenses":            ["category"],
    "amortization":        ["asset_name"],
    "specialist_activity": ["activity_type"],
}

_FLOAT_TOLERANCE = 0.01


def _fetch_db_table(supabase, table_name: str) -> pd.DataFrame:
    """Load all rows from a Supabase table using pagination."""
    rows, offset, page = [], 0, 500
    while True:
        resp = supabase.table(table_name).select("*").range(offset, offset + page - 1).execute()
        if not resp.data:
            break
        rows.extend(resp.data)
        if len(resp.data) < page:
            break
        offset += page
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def check_value_accuracy(anonymized_tables: dict, supabase) -> dict:
    """Compare every value in anonymized_tables against what is stored in Supabase.

    Rows are matched by source_uid, a stable hash that uniquely identifies each record.
    """
    total_checks = 0
    mismatches = []

    for table_name in DB_TABLES:
        df_src = anonymized_tables.get(table_name, pd.DataFrame())
        if df_src.empty or "source_uid" not in df_src.columns:
            continue

        df_db = _fetch_db_table(supabase, table_name)
        if df_db.empty or "source_uid" not in df_db.columns:
            mismatches.append(f"{table_name}: не вдалось зчитати з БД")
            continue

        db_index = df_db.set_index("source_uid")
        numeric_cols = _NUMERIC_FIELDS.get(table_name, [])
        text_cols = _TEXT_FIELDS.get(table_name, [])

        for _, src_row in df_src.iterrows():
            uid = src_row.get("source_uid")
            if uid not in db_index.index:
                mismatches.append(f"{table_name} [{uid[:8]}...]: рядок відсутній у БД")
                continue

            db_row = db_index.loc[uid]

            for col in numeric_cols:
                total_checks += 1
                src_val = src_row.get(col)
                db_val = db_row.get(col)

                src_none = src_val is None or (isinstance(src_val, float) and pd.isna(src_val))
                db_none = db_val is None or (isinstance(db_val, float) and pd.isna(db_val))

                if src_none and db_none:
                    continue
                if src_none != db_none:
                    mismatches.append(
                        f"{table_name}.{col} [{uid[:8]}...]: "
                        f"очікувано {'NULL' if src_none else src_val}, "
                        f"у БД {'NULL' if db_none else db_val}"
                    )
                    continue
                try:
                    if abs(float(src_val) - float(db_val)) > _FLOAT_TOLERANCE:
                        mismatches.append(
                            f"{table_name}.{col} [{uid[:8]}...]: "
                            f"очікувано {src_val}, у БД {db_val}"
                        )
                except (TypeError, ValueError):
                    mismatches.append(
                        f"{table_name}.{col} [{uid[:8]}...]: не вдалось порівняти ({src_val} vs {db_val})"
                    )

            for col in text_cols:
                total_checks += 1
                src_val = src_row.get(col)
                db_val = db_row.get(col)

                src_none = src_val is None or (not isinstance(src_val, str) and pd.isna(src_val))
                db_none = db_val is None

                if src_none and db_none:
                    continue
                src_str = str(src_val).strip().lower() if not src_none else None
                db_str = str(db_val).strip().lower() if not db_none else None
                if src_str != db_str:
                    mismatches.append(
                        f"{table_name}.{col} [{uid[:8]}...]: "
                        f"очікувано '{src_str}', у БД '{db_str}'"
                    )

    score = ((total_checks - len(mismatches)) / total_checks * 100) if total_checks else 100.0
    detail = f"{len(mismatches)} розбіжностей з {total_checks} перевірок"
    if mismatches:
        detail += ": " + "; ".join(mismatches[:10])
        if len(mismatches) > 10:
            detail += f" ... (+{len(mismatches) - 10} ще)"
    return {"score": round(score, 1), "detail": detail}


def check_sync_time(timings: dict) -> dict:
    total = timings.get("total", 0)
    ok = total < SYNC_TIME_TARGET
    detail = " | ".join(f"{k}: {v:.1f}с" for k, v in timings.items())
    return {"score": total, "ok": ok, "detail": detail}


def save_results_csv(results: dict, run_ts: datetime) -> Path:
    """Write test results to a CSV file in tests/results/."""
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    filename = out_dir / f"sync_{run_ts.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    m1 = results["structural_integrity"]
    m2 = results["data_types"]
    m3 = results["anonymization"]
    m4 = results["irreversibility"]
    m5 = results["record_loss"]
    m6 = results["value_accuracy"]
    m7 = results["sync_time"]
    ts = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        {
            "timestamp": ts, "metric": "Структурна цілісність",
            "result":    f"{m1['score']}%" if m1["score"] is not None else "н/д",
            "target":    "100%",
            "passed":    "TRUE" if (m1["score"] is None or m1["score"] == 100.0) else "FALSE",
            "detail":    m1["detail"],
        },
        {
            "timestamp": ts, "metric": "Коректність типів даних",
            "result":    f"{m2['score']}%",
            "target":    "100%",
            "passed":    "TRUE" if m2["score"] == 100.0 else "FALSE",
            "detail":    m2["detail"],
        },
        {
            "timestamp": ts, "metric": "Анонімізація полів",
            "result":    f"{m3['score']}%" if m3["score"] is not None else "н/д",
            "target":    "100%",
            "passed":    "TRUE" if (m3["score"] is None or m3["score"] == 100.0) else "FALSE",
            "detail":    m3["detail"],
        },
        {
            "timestamp": ts, "metric": "Незворотність анонімізації",
            "result":    "Так" if m4["score"] else "НІ",
            "target":    "Так",
            "passed":    "TRUE" if m4["score"] else "FALSE",
            "detail":    m4["detail"],
        },
        {
            "timestamp": ts, "metric": "Втрата/дублювання записів",
            "result":    f"{100.0 - m5['score']:.1f}%",
            "target":    "0%",
            "passed":    "TRUE" if m5["score"] == 100.0 else "FALSE",
            "detail":    m5["detail"],
        },
        {
            "timestamp": ts, "metric": "Точність значень у БД",
            "result":    f"{m6['score']}%",
            "target":    "100%",
            "passed":    "TRUE" if m6["score"] == 100.0 else "FALSE",
            "detail":    m6["detail"],
        },
        {
            "timestamp": ts, "metric": "Час синхронізації (с)",
            "result":    f"{m7['score']:.1f}",
            "target":    f"< {SYNC_TIME_TARGET:.0f}",
            "passed":    "TRUE" if m7["ok"] else "FALSE",
            "detail":    m7["detail"],
        },
    ]

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "metric", "result", "target", "passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    return filename


def _status(ok: bool) -> str:
    return "  ✓  " if ok else "  ✗  "


def print_report(results: dict):
    print()
    print("=" * 65)
    print("  РЕЗУЛЬТАТИ ТЕСТУ СИНХРОНІЗАЦІЇ")
    print("=" * 65)

    m1 = results["structural_integrity"]
    m2 = results["data_types"]
    m3 = results["anonymization"]
    m4 = results["irreversibility"]
    m5 = results["record_loss"]
    m6 = results["value_accuracy"]
    m7 = results["sync_time"]

    rows = [
        ("Структурна цілісність",       f"{m1['score']}%",        m1["score"] == 100.0 if m1["score"] is not None else True),
        ("Коректність типів даних",      f"{m2['score']}%",        m2["score"] == 100.0),
        ("Анонімізація полів",           f"{m3['score']}%" if m3["score"] is not None else "н/д", m3["score"] == 100.0 if m3["score"] is not None else True),
        ("Незворотність анонімізації",   "Так" if m4["score"] else "НІ", m4["score"]),
        ("Втрата/дублювання записів",    f"{100.0 - m5['score']:.1f}% втрат", m5["score"] == 100.0),
        ("Точність значень у БД",        f"{m6['score']}%",        m6["score"] == 100.0),
        ("Час синхронізації",            f"{m7['score']:.1f} с",   m7["ok"]),
    ]

    header = f"  {'Метрика':<35} {'Результат':>12}  {'Ціль':>10}  Статус"
    print(header)
    print("  " + "-" * 61)

    targets = ["100%", "100%", "100%", "Так", "0% втрат", "100%", f"< {SYNC_TIME_TARGET:.0f} с"]
    for (name, result, ok), target in zip(rows, targets):
        print(f"  {name:<35} {result:>12}  {target:>10}  {_status(ok)}")

    print("  " + "-" * 61)
    all_ok = all(ok for _, _, ok in rows)
    verdict = "PASSED" if all_ok else "FAILED"
    print(f"\n  Загальний результат: {verdict}")
    print()

    print("  Деталі:")
    details = [
        ("Структурна цілісність", m1["detail"]),
        ("Типи даних",            m2["detail"]),
        ("Анонімізація",          m3["detail"]),
        ("Незворотність",         m4["detail"]),
        ("Записи",                m5["detail"]),
        ("Точність значень",      m6["detail"]),
        ("Час (кроки)",           m7["detail"]),
    ]
    for label, text in details:
        print(f"  [{label}] {text}")
    print("=" * 65)



def main():
    run_ts = datetime.now()
    print("=== Тест синхронізації: Google Sheets → Supabase ===\n")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ПОМИЛКА: SUPABASE_URL або SUPABASE_SERVICE_ROLE_KEY не задані.")
        sys.exit(1)

    supabase = create_client(url, key)
    print("Supabase: підключено")

    timings = {}
    t_total = time.perf_counter()

    print("Крок 1: Читання з Google Sheets", end=" ", flush=True)
    t = time.perf_counter()
    raw_tables = load_raw_tables()
    timings["читання з Sheets"] = time.perf_counter() - t
    total_raw = sum(len(df) for df in raw_tables.values())
    print(f"{timings['читання з Sheets']:.1f} с  [{total_raw} рядків]")

    print("Крок 2: Очистка", end=" ", flush=True)
    t = time.perf_counter()
    cleaned_tables = clean_tables(raw_tables)
    timings["очистка"] = time.perf_counter() - t
    print(f"{timings['очистка']:.1f} с")

    print("Крок 3: Нормалізація", end=" ", flush=True)
    t = time.perf_counter()
    normalized_tables = normalize_tables(cleaned_tables)
    timings["нормалізація"] = time.perf_counter() - t
    print(f"{timings['нормалізація']:.1f} с")

    print("Крок 4: Валідація", end=" ", flush=True)
    t = time.perf_counter()
    try:
        validate_all(normalized_tables)
        timings["валідація"] = time.perf_counter() - t
        print(f"{timings['валідація']:.1f} с  [OK]")
    except ValueError as e:
        print(f"\nПОМИЛКА ВАЛІДАЦІЇ: {e}")
        sys.exit(1)

    input_counts = {t: len(df) for t, df in normalized_tables.items()}

    print("Крок 5: Attach identity", end=" ", flush=True)
    t = time.perf_counter()
    identified_tables = attach_identity(normalized_tables)
    timings["attach_identity"] = time.perf_counter() - t
    print(f"{timings['attach_identity']:.1f} с")

    print("Крок 6: Анонімізація", end=" ", flush=True)
    t = time.perf_counter()
    anonymized_tables = anonymize_tables(identified_tables, supabase)
    timings["анонімізація"] = time.perf_counter() - t
    print(f"{timings['анонімізація']:.1f} с")

    print("Крок 7: Запис в Supabase", end=" ", flush=True)
    t = time.perf_counter()
    touched_months = upsert_all(supabase=supabase, tables=anonymized_tables)
    timings["запис в БД"] = time.perf_counter() - t
    print(f"{timings['запис в БД']:.1f} с")

    if touched_months:
        print("Крок 8: Обчислення monthly_metrics", end=" ", flush=True)
        t = time.perf_counter()
        compute_for_updated_months(supabase, touched_months)
        timings["monthly_metrics"] = time.perf_counter() - t
        print(f"{timings['monthly_metrics']:.1f} с")

    timings["total"] = time.perf_counter() - t_total

    print("\nПеревірка метрик")
    print("  Точність значень (читання з БД)", end=" ", flush=True)
    value_accuracy = check_value_accuracy(anonymized_tables, supabase)
    print(f"перевірено {value_accuracy['detail'].split(' ')[1] if 'розбіжностей' in value_accuracy['detail'] else '?'} полів")

    results = {
        "structural_integrity": check_structural_integrity(normalized_tables),
        "data_types":           check_data_types(normalized_tables),
        "anonymization":        check_anonymization(anonymized_tables),
        "irreversibility":      check_irreversibility(anonymized_tables, supabase),
        "record_loss":          check_record_loss(input_counts, supabase),
        "value_accuracy":       value_accuracy,
        "sync_time":            check_sync_time(timings),
    }

    print_report(results)

    csv_path = save_results_csv(results, run_ts)
    print(f"\n  Результати збережено: {csv_path}")


if __name__ == "__main__":
    main()
