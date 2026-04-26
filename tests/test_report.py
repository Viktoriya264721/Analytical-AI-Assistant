import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import openai

env_path = os.getenv("ENV_PATH")
if env_path:
    from dotenv import load_dotenv
    load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage
from supabase import create_client

from agent.context import build_agent_context, resolve_names_in_text, fetch_available_months
from agent.llm import build_llm
from agent.prompt import build_full_prompt


REPORTS_DIR = Path(__file__).parent / "results" / "reports"
CSV_DIR     = Path(__file__).parent / "results" / "report_tests"
TOKENS_LOG  = CSV_DIR / "tokens_log.csv"
GEN_TIME_TARGET    = 60.0
INPUT_PRICE_PER_M  = 3.00   # $ per 1M input tokens  (Claude Sonnet 4.6)
OUTPUT_PRICE_PER_M = 15.00  # $ per 1M output tokens (Claude Sonnet 4.6)

REQUIRED_SECTIONS = [
    (r"#\s+Щомісячний фінансовий звіт",  "# Щомісячний фінансовий звіт"),
    (r"##\s*1\b",                          "## 1. Огляд фінансового стану"),
    (r"###\s*1\.1\b",                      "### 1.1 P&L звіт"),
    (r"###\s*1\.2\b",                      "### 1.2 Cash Flow звіт"),
    (r"##\s*2\b",                          "## 2. Зарплати персоналу"),
    (r"###\s*2\.1\b",                      "### 2.1 Топ-3 працівники"),
    (r"###\s*2\.2\b",                      "### 2.2 Динаміка по працівниках"),
    (r"##\s*3\b",                          "## 3. Ефективність працівників"),
    (r"###\s*3\.1\b",                      "### 3.1 Найефективніші"),
    (r"###\s*3\.2\b",                      "### 3.2 Найнижчі показники"),
    (r"##\s*4\b",                          "## 4. Розподіл послуг"),
    (r"###\s*4\.1\b",                      "### 4.1 Загальна активність"),
    (r"###\s*4\.2\b",                      "### 4.2 Аналіз по працівниках"),
    (r"###\s*4\.3\b",                      "### 4.3 Динаміка послуг"),
    (r"##\s*5\b",                          "## 5. Структура витрат"),
    (r"###\s*5\.1\b",                      "### 5.1 Топ-3 категорії витрат"),
    (r"###\s*5\.2\b",                      "### 5.2 Амортизація активів"),
    (r"###\s*5\.3\b",                      "### 5.3 Динаміка витрат"),
    (r"##\s*6\b",                          "## 6. Виплати співвласникам"),
]

_FAITHFULNESS_KEYS = [
    # P&L
    ("pnl",      "revenue"),
    ("pnl",      "ebit"),
    ("pnl",      "gross_profit"),
    ("pnl",      "owner_share_33"),
    ("pnl",      "specialist_payouts_core"),
    ("pnl",      "support_salaries"),
    ("pnl",      "total_expenses"),
    ("pnl",      "amortization"),
    # Cash Flow
    ("cashflow", "operating_cf"),
    ("cashflow", "capex"),
    ("cashflow", "opex_outflow"),
]

_HALLUCINATION_TOLERANCE = 0.02

def generate_report_timed(supabase, month: str, model_name: str | None) -> dict:
    """Generate a report with per-step timing.

    Returns a dict with:
    - raw_report    : LLM response before resolve_names (still anonymised)
    - final_report  : report with real names substituted in
    - summary_dict  : context passed to the LLM
    - timings       : {sql_context, llm_generation, resolve_names, total}
    """
    timings: dict[str, float] = {}

    t = time.perf_counter()
    summary_dict = build_agent_context(supabase, month)
    timings["sql_context"] = time.perf_counter() - t

    summary_json = json.dumps(summary_dict, ensure_ascii=False, indent=2)
    full_prompt  = build_full_prompt(summary_json)
    llm          = build_llm(model_name, temperature=0, max_tokens=4096)

    t = time.perf_counter()
    response = llm.invoke([HumanMessage(content=full_prompt)])
    timings["llm_generation"] = time.perf_counter() - t

    usage = getattr(response, "usage_metadata", None) or {}
    token_info = {
        "input_tokens":  int(usage.get("input_tokens",  0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
    }

    raw_report = response.content

    t = time.perf_counter()
    final_report = resolve_names_in_text(raw_report, supabase)
    timings["resolve_names"] = time.perf_counter() - t

    timings["total"] = sum(timings.values())
    return {
        "raw_report":   raw_report,
        "final_report": final_report,
        "summary_dict": summary_dict,
        "timings":      timings,
        "token_info":   token_info,
    }



def check_template_compliance(report: str) -> dict:
    found, missing = [], []
    for pattern, label in REQUIRED_SECTIONS:
        (found if re.search(pattern, report, re.MULTILINE) else missing).append(label)

    score = len(found) / len(REQUIRED_SECTIONS) * 100
    detail = f"{len(found)}/{len(REQUIRED_SECTIONS)} секцій присутні"
    if missing:
        detail += f"; відсутні: {', '.join(missing[:5])}"
        if len(missing) > 5:
            detail += f" (+{len(missing)-5} ще)"
    return {"score": round(score, 1), "detail": detail}



def _fmt(value: float) -> str:
    """Format a number with a space as the thousands separator."""
    return f"{int(round(value)):,}".replace(",", " ")


def _fmt_decimal(value: float) -> list[str]:
    """Return candidate decimal representations using a comma as the decimal separator."""
    candidates = []
    for decimals in (2, 1):
        rounded = round(value, decimals)
        integer_part = _fmt(int(rounded))
        frac = f"{rounded:.{decimals}f}".split(".")[1]
        candidates.append(f"{integer_part},{frac}")
    return candidates


def _value_in_report(value: float, report: str) -> bool:
    """Return True if the value (or any standard rounding of it) appears in the report text."""
    if abs(value) < 1.0:
        return True
    candidates = [
        _fmt(value),
        _fmt(round(value, -1)),
        _fmt(round(value, -2)),
        _fmt(round(value, -3)),
        str(int(round(value))),
    ]
    if value != round(value):
        candidates.extend(_fmt_decimal(value))
    return any(c in report for c in candidates)


def check_faithfulness(summary_dict: dict, final_report: str) -> dict:
    """Check that key numeric values from the context appear in the final report."""
    current  = summary_dict.get("current_month_summary", {})
    checked, missing = [], []

    for section, key in _FAITHFULNESS_KEYS:
        val = current.get(section, {}).get(key)
        if val is None:
            continue
        val = float(val)
        found = _value_in_report(val, final_report)
        checked.append((key, val, found))
        if not found:
            missing.append(f"{key}={_fmt(val)}")

    top_3_salaries = sorted(
        current.get("salaries", {}).get("top_3", {}).items(),
        key=lambda x: -x[1],
    )[:3]
    for person_id, salary in top_3_salaries:
        val = float(salary)
        found = _value_in_report(val, final_report)
        checked.append((f"зарплата:{person_id}", val, found))
        if not found:
            missing.append(f"зарплата={_fmt(val)}")

    total = len(checked)
    ok    = sum(1 for _, _, f in checked if f)
    score = ok / total * 100 if total else 100.0
    detail = f"{ok}/{total} ключових метрик знайдено у тексті"
    if missing:
        detail += f"; не знайдено: {', '.join(missing)}"
    return {"score": round(score, 1), "detail": detail}


def check_privacy(raw_report: str, supabase) -> dict:
    """Check that the raw LLM response contains no real person names from the persons table."""
    resp = supabase.table("persons").select("real_name").execute()
    real_names = [r["real_name"] for r in (resp.data or []) if r.get("real_name")]

    if not real_names:
        return {"score": True, "detail": "таблиця persons порожня — нічого перевіряти"}

    raw_lower = raw_report.lower()
    leaked = [name for name in real_names if name.strip().lower() in raw_lower]

    ok     = len(leaked) == 0
    detail = (
        "реальних імен у raw-відповіді LLM не знайдено"
        if ok else
        f"ВИТОКИ ({len(leaked)}): {', '.join(leaked)}"
    )
    return {"score": ok, "detail": detail}


def _collect_known_values(summary_dict: dict) -> set[float]:
    """Recursively collect all numeric values > 100 from summary_dict (current and historical)."""
    known: set[float] = set()

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            if abs(obj) > 100:
                known.add(float(obj))

    _walk(summary_dict)
    return known


def _extract_report_numbers(text: str) -> list[float]:
    """Extract financial numbers > 1000 from the report text.

    Handles formats: *NN NNN грн*, –NN NNN грн, NN NNN,NN грн, *NN NNN*.
    """
    numbers: list[float] = []
    seen: set[float] = set()

    def _add(raw: str) -> None:
        val_str = raw.strip().replace(" ", "").replace(",", ".")
        try:
            val = float(val_str)
            if val > 1000 and val not in seen:
                numbers.append(val)
                seen.add(val)
        except ValueError:
            pass

    for m in re.finditer(r'(\d[\d ]*\d|\d)(?:,\d+)?\s*грн', text):
        _add(m.group(0).replace("грн", ""))

    for m in re.finditer(r'\*{1,2}(\d[\d ]{3,})\*{1,2}', text):
        _add(m.group(1))

    return numbers


def check_no_hallucinations(final_report: str, summary_dict: dict) -> dict:
    """Check that large numbers in the report are present in the LLM context (within 2% tolerance)."""
    known_values   = _collect_known_values(summary_dict)
    report_numbers = _extract_report_numbers(final_report)

    if not report_numbers:
        return {"score": True, "detail": "фінансових чисел > 1000 для перевірки не знайдено"}

    unexplained: list[str] = []
    for num in report_numbers:
        matched = any(
            abs(num - abs(kv)) / max(abs(kv), 1) <= _HALLUCINATION_TOLERANCE
            for kv in known_values
        )
        if not matched:
            unexplained.append(_fmt(num))

    unexplained = list(dict.fromkeys(unexplained))

    ok = len(unexplained) == 0
    detail = (
        f"перевірено {len(report_numbers)} чисел, всі підтверджені контекстом"
        if ok else
        f"{len(unexplained)} непідтверджених: {', '.join(unexplained[:10])}"
        + (f" (+{len(unexplained)-10} ще)" if len(unexplained) > 10 else "")
    )
    return {"score": ok, "detail": detail}



def check_llm_judge(final_report: str, month: str, judge_model: str = "gpt-4o-mini") -> dict:
    """Evaluate report quality using an LLM judge on a 1–5 scale."""
    prompt = f"""Ти — власник реабілітаційного центру. Тобі щойно згенерували щомісячний фінансовий звіт за {month}.

Ось текст звіту:
{final_report}

Оціни якість цього звіту за шкалою від 1 до 5 з точки зору власника бізнесу:

5 — Відмінний звіт: чітка структура, глибокий аналіз, конкретні висновки і рекомендації — можу одразу приймати рішення.
4 — Хороший звіт: всі ключові секції присутні, аналіз здебільшого корисний, є незначні недоліки у глибині або формулюванні.
3 — Задовільний звіт: базова інформація є, але бракує аналітики, висновків або практичних рекомендацій.
2 — Слабкий звіт: значні прогалини в аналізі, поверхневі висновки, мало практичної цінності для прийняття рішень.
1 — Незадовільний звіт: відсутні ключові секції, інформація марна або структура повністю порушена.

Відповідай ТІЛЬКИ JSON-об'єктом в один рядок:
{{"score": <1-5>, "reason": "<одне коротке речення українською — чому така оцінка>"}}"""

    try:
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=judge_model,
            max_tokens=200,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content.strip())
        score = int(data["score"])
        reason = str(data.get("reason", ""))
        return {"score": score, "detail": f"{score}/5 — {reason}"}
    except Exception as e:
        return {"score": None, "detail": f"помилка судді: {e}"}


def _avg_timings(all_runs: list[dict]) -> dict[str, float]:
    if not all_runs:
        return {}
    keys = all_runs[0].keys()
    return {k: sum(r[k] for r in all_runs) / len(all_runs) for k in keys}


def _aggregate_results(all_results: list[dict]) -> dict:
    """Aggregate metrics across all runs.

    Numeric metrics (template, faithfulness) are averaged.
    Boolean metrics (privacy, hallucinations) use worst-case: one failure fails the aggregate.
    """
    n = len(all_results)

    avg_template = round(
        sum(r["template"]["score"] for r in all_results) / n, 1
    )
    avg_faith = round(
        sum(r["faithfulness"]["score"] for r in all_results) / n, 1
    )

    privacy_ok = all(r["privacy"]["score"] for r in all_results)
    privacy_worst = next(
        (r["privacy"] for r in all_results if not r["privacy"]["score"]),
        all_results[0]["privacy"],
    )

    halluc_ok = all(r["no_hallucinations"]["score"] for r in all_results)
    halluc_worst = next(
        (r["no_hallucinations"] for r in all_results if not r["no_hallucinations"]["score"]),
        all_results[0]["no_hallucinations"],
    )

    template_details = " | ".join(
        f"#{i+1}: {r['template']['score']}%" for i, r in enumerate(all_results)
    )
    faith_details = " | ".join(
        f"#{i+1}: {r['faithfulness']['score']}%" for i, r in enumerate(all_results)
    )

    return {
        "template": {
            "score": avg_template,
            "detail": f"сер. {avg_template}% ({template_details})",
        },
        "faithfulness": {
            "score": avg_faith,
            "detail": f"сер. {avg_faith}% ({faith_details})",
        },
        "privacy": {
            "score": privacy_ok,
            "detail": privacy_worst["detail"],
        },
        "no_hallucinations": {
            "score": halluc_ok,
            "detail": halluc_worst["detail"],
        },
    }


def _run_rows(result: dict, judge_result: dict, month: str, model: str,
              run_num: int, ts: str) -> list[dict]:
    """Build metric rows for a single test run."""
    m1, m2 = result["template"], result["faithfulness"]
    m3, m4 = result["privacy"],  result["no_hallucinations"]
    t       = result["timings"]["total"]
    js      = judge_result["score"]
    base = {"timestamp": ts, "month": month, "model": model, "run": run_num}
    rows = [
        {**base, "metric": "Відповідність шаблону",
         "result": f"{m1['score']}%", "target": "100%",
         "passed": "TRUE" if m1["score"] == 100.0 else "FALSE", "detail": m1["detail"]},
        {**base, "metric": "Faithfulness (ключові цифри)",
         "result": f"{m2['score']}%", "target": "100%",
         "passed": "TRUE" if m2["score"] == 100.0 else "FALSE", "detail": m2["detail"]},
        {**base, "metric": "Відсутність реальних імен (до resolve)",
         "result": "Так" if m3["score"] else "НІ — ВИТІК", "target": "Так",
         "passed": "TRUE" if m3["score"] else "FALSE", "detail": m3["detail"]},
        {**base, "metric": "Галюцинації (інформаційно)",
         "result": "Так" if m4["score"] else "НІ", "target": "Так",
         "passed": "INFO", "detail": m4["detail"]},
        {**base, "metric": "LLM-суддя (якість звіту)",
         "result": f"{js}/5" if js is not None else "н/д", "target": "≥ 4/5",
         "passed": ("TRUE" if js >= 4 else "FALSE") if js is not None else "н/д",
         "detail": judge_result["detail"]},
        {**base, "metric": "Час генерації",
         "result": f"{t:.1f} с", "target": f"< {GEN_TIME_TARGET:.0f} с",
         "passed": "TRUE" if t < GEN_TIME_TARGET else "FALSE",
         "detail": " | ".join(f"{k}: {v:.2f}с" for k, v in result["timings"].items())},
    ]
    ti   = result.get("token_info", {})
    inp  = ti.get("input_tokens",  0)
    out  = ti.get("output_tokens", 0)
    cost = (inp / 1_000_000) * INPUT_PRICE_PER_M + (out / 1_000_000) * OUTPUT_PRICE_PER_M
    rows.append({
        **base,
        "metric": "Вартість генерації",
        "result": f"${cost:.4f}",
        "target": "—",
        "passed": "INFO",
        "detail": f"input: {inp:,} токенів | output: {out:,} токенів | вартість: ${cost:.4f}",
    })
    return rows


def _summary_rows(all_results: list[dict], month: str, model: str,
                  avg_t: dict[str, float], ts: str) -> list[dict]:
    """Build summary (averaged) rows across all runs."""
    n    = len(all_results)
    agg  = _aggregate_results(all_results)
    base = {"timestamp": ts, "month": month, "model": model, "run": "summary"}

    judge_scores = [r["judge"]["score"] for r in all_results
                    if r.get("judge", {}).get("score") is not None]
    avg_judge = round(sum(judge_scores) / len(judge_scores), 1) if judge_scores else None

    t_avg = avg_t.get("total", 0)
    per_run_t = " | ".join(
        f"#{i+1}: {r['timings']['total']:.1f}с" for i, r in enumerate(all_results)
    )

    m1, m2 = agg["template"], agg["faithfulness"]
    m3, m4 = agg["privacy"],  agg["no_hallucinations"]
    rows = [
        {**base, "metric": "Відповідність шаблону",
         "result": f"{m1['score']}%", "target": "100%",
         "passed": "TRUE" if m1["score"] == 100.0 else "FALSE", "detail": m1["detail"]},
        {**base, "metric": "Faithfulness (ключові цифри)",
         "result": f"{m2['score']}%", "target": "100%",
         "passed": "TRUE" if m2["score"] == 100.0 else "FALSE", "detail": m2["detail"]},
        {**base, "metric": "Відсутність реальних імен (до resolve)",
         "result": "Так" if m3["score"] else "НІ — ВИТІК", "target": "Так",
         "passed": "TRUE" if m3["score"] else "FALSE", "detail": m3["detail"]},
        {**base, "metric": "Галюцинації (інформаційно)",
         "result": "Так" if m4["score"] else "НІ", "target": "Так",
         "passed": "INFO", "detail": m4["detail"]},
        {**base, "metric": f"Час генерації (сер. по {n} запусках)",
         "result": f"{t_avg:.1f} с", "target": f"< {GEN_TIME_TARGET:.0f} с",
         "passed": "TRUE" if t_avg < GEN_TIME_TARGET else "FALSE",
         "detail": f"сер. {t_avg:.1f} с ({per_run_t})"},
    ]
    if avg_judge is not None:
        per_run_j = " | ".join(
            f"#{i+1}: {r['judge']['score']}/5" for i, r in enumerate(all_results)
            if r.get("judge", {}).get("score") is not None
        )
        rows.append({**base, "metric": "LLM-суддя (якість звіту)",
                     "result": f"{avg_judge}/5", "target": "≥ 4/5",
                     "passed": "TRUE" if avg_judge >= 4 else "FALSE",
                     "detail": f"сер. {avg_judge}/5 ({per_run_j})"})

    token_infos = [r.get("token_info", {}) for r in all_results]
    avg_inp  = sum(ti.get("input_tokens",  0) for ti in token_infos) / n
    avg_out  = sum(ti.get("output_tokens", 0) for ti in token_infos) / n
    avg_cost = (avg_inp / 1_000_000) * INPUT_PRICE_PER_M + (avg_out / 1_000_000) * OUTPUT_PRICE_PER_M
    per_run_cost = " | ".join(
        f"#{i+1}: ${((ti.get('input_tokens',0)/1_000_000)*INPUT_PRICE_PER_M + (ti.get('output_tokens',0)/1_000_000)*OUTPUT_PRICE_PER_M):.4f}"
        for i, ti in enumerate(token_infos)
    )
    rows.append({**base, "metric": "Вартість генерації (сер.)",
                 "result": f"${avg_cost:.4f}", "target": "—", "passed": "INFO",
                 "detail": f"сер. input: {avg_inp:.0f} | сер. output: {avg_out:.0f} | {per_run_cost}"})
    return rows


def save_token_log(token_info: dict, month: str, model: str, run_ts: datetime) -> None:
    """Append one row per run to the cumulative tokens_log.csv."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "month", "model", "input_tokens", "output_tokens", "cost_usd"]
    inp  = token_info.get("input_tokens",  0)
    out  = token_info.get("output_tokens", 0)
    cost = (inp / 1_000_000) * INPUT_PRICE_PER_M + (out / 1_000_000) * OUTPUT_PRICE_PER_M
    row  = {
        "timestamp":    run_ts.strftime("%Y-%m-%d %H:%M:%S"),
        "month":        month,
        "model":        model,
        "input_tokens": inp,
        "output_tokens": out,
        "cost_usd":     round(cost, 6),
    }
    write_header = not TOKENS_LOG.exists()
    with open(TOKENS_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_combined_csv(
    all_results: list[dict],
    month: str,
    model: str,
    avg_t: dict[str, float],
    run_ts: datetime,
) -> Path:
    """Save metrics for all runs plus the aggregate summary to a single CSV."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = CSV_DIR / f"report_{month}_{run_ts.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    ts  = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    all_rows: list[dict] = []
    for i, r in enumerate(all_results):
        all_rows.extend(_run_rows(r, r["judge"], month, model, i + 1, ts))
    all_rows.extend(_summary_rows(all_results, month, model, avg_t, ts))

    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "month", "model", "run",
                           "metric", "result", "target", "passed", "detail"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    return out



def _st(ok: bool) -> str:
    return "  ✓  " if ok else "  ✗  "


def print_report(
    all_results: list[dict],
    avg_t: dict[str, float],
    month: str,
    model: str,
):
    n = len(all_results)

    agg = _aggregate_results(all_results)
    m1, m2 = agg["template"], agg["faithfulness"]
    m3, m4 = agg["privacy"],  agg["no_hallucinations"]
    total_time = avg_t.get("total", 0)

    print()
    print("=" * 72)
    print("  РЕЗУЛЬТАТИ ТЕСТУ ГЕНЕРАЦІЇ ЗВІТУ")
    print(f"  Місяць: {month}  |  Модель: {model}  |  Запусків: {n}")
    print("=" * 72)

    rows = [
        ("Відповідність шаблону",                  f"{m1['score']}%",              m1["score"] == 100.0,          "100%",           False),
        ("Faithfulness (ключові цифри)",            f"{m2['score']}%",              m2["score"] == 100.0,          "100%",           False),
        ("Відсутність реальних імен (до resolve)",  "Так" if m3["score"] else "НІ", m3["score"],                   "Так",            False),
        ("Галюцинації (інформаційно)",              "Так" if m4["score"] else "НІ", m4["score"],                   "Так",            True),
        (f"Час генерації (сер. {n} запуск.)",       f"{total_time:.1f} с",          total_time < GEN_TIME_TARGET,  f"< {GEN_TIME_TARGET:.0f} с", False),
    ]

    judge_scores = [r["judge"]["score"] for r in all_results if r.get("judge", {}).get("score") is not None]
    if judge_scores:
        avg_judge = round(sum(judge_scores) / len(judge_scores), 1)
        rows.append((
            "LLM-суддя (якість звіту, сер.)",
            f"{avg_judge}/5",
            avg_judge >= 4,
            "≥ 4/5",
            False,
        ))

    print(f"\n  {'Метрика':<44} {'Результат':>10}  {'Ціль':>10}  Статус")
    print("  " + "─" * 68)
    for name, result, ok, target, info in rows:
        status = "  ℹ  " if info else _st(ok)
        print(f"  {name:<44} {result:>10}  {target:>10}  {status}")
    print("  " + "─" * 68)

    all_ok  = all(ok for _, _, ok, _, info in rows if not info)
    verdict = "PASSED" if all_ok else "FAILED"
    print(f"\n  Загальний результат: {verdict}")

    step_labels = {
        "sql_context":    "SQL контекст",
        "llm_generation": "LLM генерація",
        "resolve_names":  "Resolve імен",
        "total":          "ЗАГАЛОМ",
    }
    print("\n  Час генерації (середнє по кроках):")
    for k, v in avg_t.items():
        print(f"    {step_labels.get(k, k):<22} {v:.2f} с")

    print("\n  Деталі метрик:")
    details = [
        ("Шаблон",        m1["detail"]),
        ("Faithfulness",  m2["detail"]),
        ("Приватність",   m3["detail"]),
        ("Галюцинації",   m4["detail"]),
    ]
    for i, r in enumerate(all_results):
        if r.get("judge", {}).get("score") is not None:
            details.append((f"Суддя #{i+1}", r["judge"]["detail"]))
    for label, text in details:
        print(f"  [{label}] {text}")
    print("=" * 72)



def main():
    parser = argparse.ArgumentParser(description="Тест генерації фінансового звіту")
    parser.add_argument("--month", default=None,
                        help="Місяць у форматі YYYY-MM (за замовч. — останній доступний)")
    parser.add_argument("--runs",  type=int, default=1,
                        help="К-сть запусків для усереднення часу (за замовч. 1)")
    parser.add_argument("--model", default=None,
                        help="Назва моделі: haiku / sonnet / mistral:7b-q4 / ...")
    parser.add_argument("--judge-model", default="gpt-4o-mini",
                        help="Модель-суддя: gpt-4o-mini / gpt-4o (за замовч. gpt-4o-mini)")
    args = parser.parse_args()

    run_ts = datetime.now()
    print("=== Тест генерації звіту ===\n")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ПОМИЛКА: SUPABASE_URL або SUPABASE_SERVICE_ROLE_KEY не задані.")
        sys.exit(1)

    supabase = create_client(url, key)
    print("Supabase: підключено")

    if args.month:
        month = args.month
    else:
        available = fetch_available_months(supabase)
        if not available:
            print("ПОМИЛКА: monthly_metrics порожня — спочатку запустіть синхронізацію.")
            sys.exit(1)
        month = available[0]

    model       = args.model or "sonnet"
    judge_model = args.judge_model
    runs        = max(1, args.runs)
    print(f"Місяць: {month}  |  Модель: {model}  |  Суддя: {judge_model}  |  Запусків: {runs}\n")

    all_results: list[dict] = []
    all_timings: list[dict] = []

    for i in range(runs):
        prefix = f"[{i+1}/{runs}] " if runs > 1 else ""
        print(f"{prefix}Генерація звіту...", end=" ", flush=True)

        gen = generate_report_timed(supabase, month, model)
        print(f"готово ({gen['timings']['total']:.1f} с)")

        raw_report   = gen["raw_report"]
        final_report = gen["final_report"]
        summary_dict = gen["summary_dict"]

        print(f"{prefix}Перевірка метрик...", end=" ", flush=True)
        result = {
            "template":          check_template_compliance(final_report),
            "faithfulness":      check_faithfulness(summary_dict, final_report),
            "privacy":           check_privacy(raw_report, supabase),
            "no_hallucinations": check_no_hallucinations(final_report, summary_dict),
            "timings":           gen["timings"],
            "token_info":        gen["token_info"],
            "final_report":      final_report,
        }
        print("готово")

        print(f"{prefix}LLM-суддя ({judge_model})...", end=" ", flush=True)
        judge_result = check_llm_judge(final_report, month, judge_model)
        print(f"готово ({judge_result['detail']})")

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_txt = REPORTS_DIR / f"report_{month}_{run_ts.strftime('%Y-%m-%d_%H-%M-%S')}_run{i+1}.md"
        report_txt.write_text(final_report, encoding="utf-8")

        all_results.append({**result, "judge": judge_result})
        all_timings.append(gen["timings"])
        save_token_log(gen["token_info"], month, model, run_ts)

    avg_t = _avg_timings(all_timings)

    csv_path = save_combined_csv(all_results, month, model, avg_t, run_ts)

    print_report(all_results, avg_t, month, model)
    print(f"\n  CSV: {csv_path}")
    print(f"  Тексти звітів: tests/results/reports/report_{month}_..._runN.md")


if __name__ == "__main__":
    main()
