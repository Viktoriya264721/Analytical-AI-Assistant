from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

env_path = os.getenv("ENV_PATH")
if env_path:
    from dotenv import load_dotenv
    load_dotenv(env_path)

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
import openai
from langchain_core.messages import AIMessage, HumanMessage
from supabase import create_client

from agent.context import fetch_available_months, resolve_names_in_text
from agent.graph import build_graph

GROUND_TRUTH_FILE  = Path(__file__).parent / "chat_ground_truth.json"
RESULTS_DIR        = Path(__file__).parent / "results"
CHAT_TOKENS_LOG    = RESULTS_DIR / "chat" / "chat_tokens_log.csv"

INPUT_PRICE_PER_M  = 3.00 
OUTPUT_PRICE_PER_M = 15.00 
REFUSAL_WORDS = [
    "не можу", "відмовляю", "не маю", "вибачте", "вибач",
    "не в моїй", "поза моєю", "не відповідаю", "не відповідатиму",
    "лише фінансових", "не здатен", "не передбачено", "не призначений",
    "моя компетенція", "не маю доступу", "не буду",
    "не знайдено", "немає даних", "недоступно",
]

def _normalize_number(s: str) -> float:
    """Normalise a number string to float.

    Handles space-separated thousands (``"8 100"`` → ``8100.0``) and
    comma decimals (``"32,7"`` → ``32.7``).
    """
    return float(s.replace(" ", "").replace(",", "."))


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from *text*, handling space-separated thousands."""
    pattern = r"\d{1,3}(?:\s\d{3})+(?:[,\.]\d+)?|\d+(?:[,\.]\d+)?"
    results = []
    for m in re.findall(pattern, text):
        try:
            results.append(_normalize_number(m))
        except ValueError:
            pass
    return results


def _approx_equal(a: float, b: float, tol: float = 0.005, abs_cap: float = 100.0) -> bool:
    """Return ``True`` when *a* ≈ *b* within a relative tolerance capped absolutely.

    Default: 0.5 % relative, minimum ±1, maximum ±100.
    For large financial figures this keeps the tolerance tight (e.g. 164 200 → ±100)
    while still covering integer-rounding artifacts on small values.
    """
    return abs(a - b) <= max(1.0, min(abs(a) * tol, abs_cap))



def extract_tool_info(messages: list) -> tuple[list[str], list[str]]:
    """Extract tool call names and result strings from a message list.

    Args:
        messages: LangGraph message history.

    Returns:
        A tuple of ``(tool_call_names, tool_result_contents)``.
    """
    tool_calls: list[str] = []
    tool_results: list[str] = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                if name:
                    tool_calls.append(name)
        if msg.__class__.__name__ == "ToolMessage":
            content = getattr(msg, "content", "") or ""
            tool_results.append(content)

    return tool_calls, tool_results


def extract_tokens(messages: list) -> dict:
    """Sum input/output tokens across all AIMessage objects in a LangGraph result.

    In a ReAct loop the LLM is called multiple times; each call produces an
    AIMessage with its own usage_metadata.  Summing them gives the total cost
    for the full interaction.
    """
    inp = sum(
        (getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0)
        for m in messages if isinstance(m, AIMessage)
    )
    out = sum(
        (getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
        for m in messages if isinstance(m, AIMessage)
    )
    return {"input_tokens": inp, "output_tokens": out}



def score_numeric(expected: str, response: str) -> bool:
    """Return ``True`` when the expected number appears in *response* within tolerance (0.5 %, max ±100)."""
    try:
        exp_val = _normalize_number(expected.split(";")[0].strip())
    except ValueError:
        return False
    nums = _extract_numbers(response)
    return any(_approx_equal(n, exp_val) for n in nums)


def score_contains(expected: str, response: str) -> bool:
    """Return ``True`` when all semicolon-separated parts appear in *response*.

    Numeric parts are matched with the same tolerance as score_numeric (0.5 %, max ±100).
    String parts are matched as case-insensitive substrings.
    """
    parts = [p.strip() for p in expected.split(";") if p.strip()]
    response_lower = response.lower()
    response_nums = _extract_numbers(response)

    for part in parts:
        nums_in_part = _extract_numbers(part)
        if nums_in_part:
            if not any(_approx_equal(n, nums_in_part[0]) for n in response_nums):
                return False
        else:
            if part.lower() not in response_lower:
                return False
    return True


def score_refusal(tool_calls: list[str], response: str) -> bool:
    """Return ``True`` when the agent made no tool calls and used a refusal phrase."""
    if len(tool_calls) > 0:
        return False
    response_lower = response.lower()
    return any(w in response_lower for w in REFUSAL_WORDS)


def score_tool_selection(actual: list[str], expected: list[str]) -> dict[str, float]:
    """Return recall, precision, and F1 for tool selection.

    When *expected* is empty (adversarial cases), all metrics are 1.0 only
    if *actual* is also empty.
    """
    if not expected:
        perfect = 1.0 if not actual else 0.0
        return {"recall": perfect, "precision": perfect, "f1": perfect}
    actual_set = set(actual)
    expected_set = set(expected)
    matched = actual_set & expected_set
    recall = len(matched) / len(expected_set)
    precision = len(matched) / len(actual_set) if actual_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "f1": round(f1, 3),
    }


def score_tool_success(tool_results: list[str]) -> float:
    """Return the fraction of tool results that contain no error message."""
    if not tool_results:
        return 1.0
    errors = sum(
        1 for r in tool_results
        if '"error"' in r.lower() or "error" in r.lower()[:30]
    )
    return (len(tool_results) - errors) / len(tool_results)


def score_faithfulness(response: str, tool_results: list[str]) -> float:
    """Return the fraction of response numbers (>100) confirmed by tool data (±1 %).

    Year-like values (2020–2030) are excluded from the check.
    """
    response_nums = [
        n for n in _extract_numbers(response)
        if n > 100 and not (2020 <= n <= 2030)
    ]
    if not response_nums:
        return 1.0

    all_tool_nums: list[float] = []
    for tr in tool_results:
        all_tool_nums.extend(_extract_numbers(tr))

    if not all_tool_nums:
        return 0.0

    confirmed = sum(
        1 for n in response_nums
        if any(_approx_equal(n, tn) for tn in all_tool_nums)
    )
    return confirmed / len(response_nums)


def llm_judge(question: str, expected: str, actual: str, judge_model: str = "gpt-4o") -> tuple[int, str]:
    """Score the agent's answer on a 1–5 scale using an LLM as judge.

    The judge acts as a rehabilitation center owner evaluating the assistant's answer.
    Supports OpenAI models (gpt-4o, gpt-4o-mini) and Anthropic models (claude-opus-4-7).

    Args:
        question: Original user question in Ukrainian.
        expected: Ground-truth expected answer.
        actual: Agent's actual response.
        judge_model: Model to use as judge.

    Returns:
        A tuple of ``(score, reason)`` where *score* is 1–5.
    """
    prompt = f"""Ти — власник реабілітаційного центру. Ти щойно поставив запитання своєму ШІ-фінансовому асистенту і хочеш оцінити, наскільки корисною і точною була його відповідь.

Твоє запитання: {question}

Відповідь асистента: {actual}

Для довідки, правильна відповідь має містити: {expected}

Оціни відповідь асистента за шкалою від 1 до 5, з точки зору власника бізнесу:

5 — Відповідь повністю правильна і корисна: цифри точні, інформація вичерпна, я можу одразу приймати рішення на її основі.
4 — Відповідь здебільшого правильна: є незначні неточності у формулюванні, але головна суть і цифри вірні.
3 — Відповідь частково корисна: ключова інформація присутня, але є пропуски або незначні помилки у даних.
2 — Відповідь переважно неправильна: головна думка хибна або відсутні критично важливі цифри.
1 — Відповідь марна: неправильна, відмова відповідати або повністю не по темі.

Відповідай ТІЛЬКИ JSON-об'єктом в один рядок:
{{"score": <1-5>, "reason": "<одне коротке речення українською — чому така оцінка>"}}"""

    try:
        if judge_model.startswith("gpt"):
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=judge_model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
        else:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=judge_model,
                max_tokens=200,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()

        data = json.loads(text)
        return int(data["score"]), str(data.get("reason", ""))
    except Exception:
        return 1, "failed to parse judge response"



ERROR_TYPES = ["wrong_tool", "hallucination", "missing_data", "wrong_reasoning", "refused_to_answer"]


def classify_error(
    question: str,
    expected: str,
    actual: str,
    tools_expected: list[str],
    tools_called: list[str],
    judge_model: str = "gpt-4o",
) -> str:
    """Classify why the agent's answer was wrong using an LLM judge.

    Returns one of: wrong_tool | hallucination | missing_data |
    wrong_reasoning | refused_to_answer | ok
    """
    prompt = f"""Ти — експерт з оцінки якості відповідей ШІ-асистентів. Проаналізуй, чому відповідь асистента є невірною.

Запитання: {question}
Очікувана відповідь: {expected}
Реальна відповідь асистента: {actual}
Очікувані інструменти: {tools_expected}
Викликані інструменти: {tools_called}

Визнач ОДНУ причину помилки з наступних категорій:
- wrong_tool: асистент викликав неправильний інструмент або не той інструмент
- hallucination: асистент вигадав дані, яких немає в результатах інструментів
- missing_data: дані є в БД, але асистент не зміг їх отримати або відповів "немає даних"
- wrong_reasoning: дані отримано правильно, але асистент зробив помилковий розрахунок або висновок
- refused_to_answer: асистент відмовився відповідати на легітимне запитання

Відповідай ТІЛЬКИ JSON-об'єктом в один рядок:
{{"error_type": "<одна з категорій вище>", "reason": "<одне коротке речення українською>"}}"""

    try:
        if judge_model.startswith("gpt"):
            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model=judge_model,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
        else:
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=judge_model,
                max_tokens=150,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()

        data = json.loads(text)
        error_type = data.get("error_type", "wrong_reasoning")
        return error_type if error_type in ERROR_TYPES else "wrong_reasoning"
    except Exception:
        return "wrong_reasoning"


def run_test_case(
    graph,
    case: dict,
    available_months: list[str],
    latest_month: str,
    supabase,
    judge_model: str = "gpt-4o",
) -> dict:
    """Run a single test case through the agent and compute all scores.

    Args:
        graph: Compiled LangGraph agent.
        case: Test case dict from the ground-truth file.
        available_months: List of month strings available in the database.
        latest_month: Most recent available month string.
        supabase: Authenticated Supabase client.

    Returns:
        Dict with question metadata, scores, and pass/fail status.
    """
    session_id = str(uuid4())
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
    initial_state = {
        "messages": [HumanMessage(content=case["question"])],
        "available_months": available_months,
        "latest_month": latest_month,
    }

    t0 = perf_counter()
    for attempt in range(3):
        try:
            result = graph.invoke(initial_state, config=config)
            break
        except Exception as exc:
            if attempt == 2:
                raise
            print(f"         [retry {attempt + 1}/2] {exc}")
            time.sleep(2)
    elapsed = round(perf_counter() - t0, 2)

    token_info = extract_tokens(result["messages"])

    raw = result["messages"][-1].content if result["messages"] else ""
    if isinstance(raw, list):
        raw = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    response_raw = raw
    response = resolve_names_in_text(response_raw, supabase)

    tool_calls, tool_results = extract_tool_info(result["messages"])

    check_type = case["check_type"]
    expected = case["expected_answer"]
    expected_tools = case.get("expected_tools", [])

    correctness_score = None
    correctness_method = check_type

    if check_type == "numeric":
        correctness_score = 1.0 if score_numeric(expected, response) else 0.0
    elif check_type == "contains":
        correctness_score = 1.0 if score_contains(expected, response) else 0.0
    elif check_type == "refusal":
        correctness_score = 1.0 if score_refusal(tool_calls, response) else 0.0
    elif check_type == "text":
        judge_score, judge_reason = llm_judge(case["question"], expected, response, judge_model)
        correctness_score = judge_score / 5.0
        correctness_method = f"llm_judge:{judge_score}/5 — {judge_reason}"

    tool_metrics = score_tool_selection(tool_calls, expected_tools)
    tool_succ = score_tool_success(tool_results)
    faith = score_faithfulness(response, tool_results)

    corr_rounded = round(correctness_score, 3) if correctness_score is not None else None
    error_type = ""
    if corr_rounded is not None and corr_rounded < 1.0:
        error_type = classify_error(
            case["question"], expected, response,
            expected_tools, tool_calls, judge_model,
        )

    return {
        "question_id": case["id"],
        "difficulty": case.get("difficulty", ""),
        "category": case["category"],
        "question": case["question"],
        "expected_answer": expected,
        "actual_response": response[:2000],
        "tools_expected": "|".join(expected_tools),
        "tools_called": "|".join(tool_calls),
        "tool_recall": tool_metrics["recall"],
        "tool_precision": tool_metrics["precision"],
        "tool_f1": tool_metrics["f1"],
        "tool_success_rate": round(tool_succ, 3),
        "correctness_score": corr_rounded if corr_rounded is not None else "",
        "correctness_method": correctness_method,
        "faithfulness_score": round(faith, 3),
        "error_type": error_type,
        "elapsed_sec": elapsed,
        "input_tokens": token_info["input_tokens"],
        "output_tokens": token_info["output_tokens"],
        "cost_usd": round(
            (token_info["input_tokens"] / 1_000_000) * INPUT_PRICE_PER_M
            + (token_info["output_tokens"] / 1_000_000) * OUTPUT_PRICE_PER_M,
            6,
        ),
    }


def print_case_result(case: dict, res: dict, idx: int, total: int):
    """Print a one-line summary of a single test-case result."""
    corr = res["correctness_score"]
    faith = res["faithfulness_score"]
    elapsed = res["elapsed_sec"]
    tools_str = res["tools_called"] or "—"
    print(
        f"  [{idx:>3}/{total}] id={res['question_id']:>3} [{res.get('difficulty','?'):<6}] "
        f"cat={res['category']:<12} "
        f"corr={corr:<6} faith={faith:<6} "
        f"f1={res['tool_f1']:<5} "
        f"t={elapsed:.1f}s  tools=[{tools_str}]"
    )
    if corr == "" or (isinstance(corr, float) and corr < 1.0):
        print(f"         Q: {res['question'][:70]}")
        print(f"         Expected : {res['expected_answer'][:80]}")
        print(f"         Got      : {res['actual_response'][:300]}")
        if res.get("error_type"):
            print(f"         Error type: {res['error_type']}")



def print_summary(results: list[dict]):
    """Print an aggregate summary of all test results by category and difficulty."""
    total = len(results)
    by_cat: dict[str, list] = {}
    by_diff: dict[str, list] = {}
    error_counts: dict[str, int] = {}

    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
        by_diff.setdefault(r.get("difficulty", "?"), []).append(r)
        et = r.get("error_type", "")
        if et:
            error_counts[et] = error_counts.get(et, 0) + 1

    corr_vals = [r["correctness_score"] for r in results if r["correctness_score"] != ""]
    avg_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0
    avg_faith = sum(r["faithfulness_score"] for r in results) / total if total else 0
    avg_recall = sum(r["tool_recall"] for r in results) / total if total else 0
    avg_prec = sum(r["tool_precision"] for r in results) / total if total else 0
    avg_f1 = sum(r["tool_f1"] for r in results) / total if total else 0
    avg_succ = sum(r["tool_success_rate"] for r in results) / total if total else 0
    avg_time = sum(r["elapsed_sec"] for r in results) / total if total else 0

    print()
    print("=" * 70)
    print("  CHAT TEST SUMMARY")
    print("=" * 70)
    print(f"  Total cases       : {total}")
    print(f"  Correctness (avg) : {avg_corr:.3f}")
    print(f"  Faithfulness (avg): {avg_faith:.3f}")
    print(f"  Tool Recall       : {avg_recall:.3f}")
    print(f"  Tool Precision    : {avg_prec:.3f}")
    print(f"  Tool F1           : {avg_f1:.3f}")
    print(f"  Tool Success      : {avg_succ:.3f}")
    print(f"  Avg response time : {avg_time:.1f} s")

    print()
    print("  By category:")
    for cat, rows in sorted(by_cat.items()):
        cv = [r["correctness_score"] for r in rows if r["correctness_score"] != ""]
        cat_corr = sum(cv) / len(cv) if cv else 0
        print(f"    {cat:<15} n={len(rows):>2}  corr={cat_corr:.3f}")

    print()
    print("  By difficulty:")
    for diff in ("easy", "medium", "hard"):
        rows = by_diff.get(diff, [])
        if not rows:
            continue
        cv = [r["correctness_score"] for r in rows if r["correctness_score"] != ""]
        diff_corr = sum(cv) / len(cv) if cv else 0
        print(f"    {diff:<8} n={len(rows):>2}  corr={diff_corr:.3f}")

    if error_counts:
        print()
        print("  Error taxonomy (wrong cases):")
        for et, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {et:<25} {cnt}")

    print("=" * 70)



def parse_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Chat agent evaluation harness")
    p.add_argument("--model", default="sonnet",
                   help="Agent model: haiku | sonnet | mistral:7b-q4 (default: sonnet)")
    p.add_argument("--judge-model", default="gpt-4o",
                   help="Judge model: gpt-4o | gpt-4o-mini | claude-opus-4-7 (default: gpt-4o)")
    p.add_argument("--category", default=None,
                   help="Filter by category: financial | people | adversarial | ...")
    p.add_argument("--id", type=int, default=None,
                   help="Run a single test case by id")
    p.add_argument("--ids", default=None,
                   help="Comma-separated list of ids to run, e.g. --ids 1,5,12")
    return p.parse_args()


def main():
    """Run the full evaluation suite and save results."""
    args = parse_args()
    run_ts = datetime.now()

    judge_model = args.judge_model
    print(f"=== Chat agent evaluation (agent: {args.model} | judge: {judge_model}) ===\n")

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is not set.")
        sys.exit(1)

    if not GROUND_TRUTH_FILE.exists():
        print(f"ERROR: {GROUND_TRUTH_FILE} not found.")
        sys.exit(1)

    print("Connecting to Supabase...", end=" ", flush=True)
    supabase = create_client(url, key)
    print("OK")

    print("Loading available months...", end=" ", flush=True)
    available_months = fetch_available_months(supabase)
    latest_month = available_months[0] if available_months else ""
    print(f"{len(available_months)} months, latest: {latest_month}")

    print("Building agent graph...", end=" ", flush=True)
    graph = build_graph(supabase, args.model)
    print("OK\n")

    with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
        all_cases = json.load(f)

    if args.id is not None:
        cases = [c for c in all_cases if c["id"] == args.id]
        if not cases:
            print(f"ERROR: test case id={args.id} not found.")
            sys.exit(1)
    elif args.ids is not None:
        id_set = {int(x.strip()) for x in args.ids.split(",")}
        cases = [c for c in all_cases if c["id"] in id_set]
        if not cases:
            print(f"ERROR: none of the ids {args.ids} found.")
            sys.exit(1)
    elif args.category:
        cases = [c for c in all_cases if c["category"] == args.category]
        if not cases:
            print(f"ERROR: category '{args.category}' not found.")
            sys.exit(1)
    else:
        cases = all_cases

    print(f"Running {len(cases)} test cases...\n")

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / f"chat_{run_ts.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    fieldnames = [
        "timestamp", "question_id", "difficulty", "category", "question",
        "expected_answer", "actual_response",
        "tools_expected", "tools_called",
        "tool_recall", "tool_precision", "tool_f1",
        "tool_success_rate", "correctness_score", "correctness_method",
        "faithfulness_score", "error_type", "elapsed_sec",
        "input_tokens", "output_tokens", "cost_usd",
    ]
    ts = run_ts.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Writing results live to: {csv_path}\n")

    results = []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, case in enumerate(cases, 1):
            print(f"  Test {i}/{len(cases)} (id={case['id']}): {case['question'][:60]}")
            try:
                res = run_test_case(graph, case, available_months, latest_month, supabase, judge_model)
            except Exception as exc:
                res = {
                    "question_id": case["id"],
                    "difficulty": case.get("difficulty", ""),
                    "category": case["category"],
                    "question": case["question"],
                    "expected_answer": case["expected_answer"],
                    "actual_response": f"ERROR: {exc}",
                    "tools_expected": "|".join(case.get("expected_tools", [])),
                    "tools_called": "",
                    "tool_recall": 0.0,
                    "tool_precision": 0.0,
                    "tool_f1": 0.0,
                    "tool_success_rate": 0.0,
                    "correctness_score": 0.0,
                    "correctness_method": "error",
                    "faithfulness_score": 0.0,
                    "error_type": "missing_data",
                    "elapsed_sec": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            results.append(res)
            writer.writerow({"timestamp": ts, **res})
            f.flush()
            print_case_result(case, res, i, len(cases))

    print_summary(results)
    print(f"\n  Results saved: {csv_path}")


if __name__ == "__main__":
    main()
