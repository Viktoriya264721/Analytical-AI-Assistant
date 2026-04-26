from __future__ import annotations

import argparse
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

from langchain_core.messages import AIMessage
from supabase import create_client

from agent.context import fetch_available_months
from agent.graph import build_graph, generate_welcome, generate_follow_up_questions

RESULTS_DIR       = Path(__file__).parent / "results" / "chat"
TOKENS_LOG        = RESULTS_DIR / "topic_tokens_log.csv"

INPUT_PRICE_PER_M  = 3.00
OUTPUT_PRICE_PER_M = 15.00


def extract_tokens(messages: list) -> dict:
    inp = sum(
        (getattr(m, "usage_metadata", None) or {}).get("input_tokens", 0)
        for m in messages if isinstance(m, AIMessage)
    )
    out = sum(
        (getattr(m, "usage_metadata", None) or {}).get("output_tokens", 0)
        for m in messages if isinstance(m, AIMessage)
    )
    return {"input_tokens": inp, "output_tokens": out}


def _cost(token_info: dict) -> float:
    return round(
        (token_info["input_tokens"] / 1_000_000) * INPUT_PRICE_PER_M
        + (token_info["output_tokens"] / 1_000_000) * OUTPUT_PRICE_PER_M,
        6,
    )


def _append_row(row: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "call_type", "model", "input_tokens", "output_tokens", "cost_usd"]
    write_header = not TOKENS_LOG.exists()
    with open(TOKENS_LOG, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _measure_welcome(graph, available_months, latest_month, model, ts) -> dict:
    """Call generate_welcome and return token info."""
    import uuid
    from langchain_core.messages import HumanMessage
    from agent.graph import _WELCOME_PROMPT_TEMPLATE

    months_str = ", ".join(available_months) if available_months else latest_month
    prompt = _WELCOME_PROMPT_TEMPLATE.format(available_months=months_str)
    ephemeral = f"welcome-cost-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": ephemeral}, "recursion_limit": 50}
    state = {
        "messages": [HumanMessage(content=prompt)],
        "available_months": available_months,
        "latest_month": latest_month,
    }
    result = graph.invoke(state, config=config)
    ti = extract_tokens(result["messages"])
    return {
        "timestamp": ts,
        "call_type": "welcome",
        "model": model,
        "input_tokens": ti["input_tokens"],
        "output_tokens": ti["output_tokens"],
        "cost_usd": _cost(ti),
    }


def _measure_followup(graph, available_months, latest_month, model, ts) -> dict:
    """Call generate_follow_up_questions and return token info."""
    import uuid
    from langchain_core.messages import HumanMessage
    from agent.graph import _FOLLOW_UP_PROMPT_TEMPLATE

    recent_text = "USER: скільки заробили в листопаді?\nASSISTANT: У листопаді виручка склала 217 900 грн."
    months_str = ", ".join(available_months) if available_months else latest_month
    prompt = _FOLLOW_UP_PROMPT_TEMPLATE.format(
        recent_messages=recent_text,
        available_months=months_str,
    )
    ephemeral = f"followup-cost-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": ephemeral}, "recursion_limit": 50}
    state = {
        "messages": [HumanMessage(content=prompt)],
        "available_months": available_months,
        "latest_month": latest_month,
    }
    result = graph.invoke(state, config=config)
    ti = extract_tokens(result["messages"])
    return {
        "timestamp": ts,
        "call_type": "follow_up",
        "model": model,
        "input_tokens": ti["input_tokens"],
        "output_tokens": ti["output_tokens"],
        "cost_usd": _cost(ti),
    }


def main():
    parser = argparse.ArgumentParser(description="Measure topic-suggestion token cost")
    parser.add_argument("--runs",  type=int, default=3, help="Runs per call type (default: 3)")
    parser.add_argument("--model", default="sonnet", help="Agent model (default: sonnet)")
    args = parser.parse_args()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("ERROR: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY not set.")
        sys.exit(1)

    supabase = create_client(url, key)
    available_months = fetch_available_months(supabase)
    latest_month = available_months[0] if available_months else ""
    graph = build_graph(supabase, args.model)

    print(f"=== Topic cost measurement (model: {args.model}, runs: {args.runs}) ===\n")

    welcome_costs, followup_costs = [], []

    for i in range(args.runs):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{i+1}/{args.runs}] welcome...", end=" ", flush=True)
        row = _measure_welcome(graph, available_months, latest_month, args.model, ts)
        _append_row(row)
        welcome_costs.append(row["cost_usd"])
        print(f"in={row['input_tokens']} out={row['output_tokens']} cost=${row['cost_usd']:.4f}")

        print(f"[{i+1}/{args.runs}] follow_up...", end=" ", flush=True)
        row = _measure_followup(graph, available_months, latest_month, args.model, ts)
        _append_row(row)
        followup_costs.append(row["cost_usd"])
        print(f"in={row['input_tokens']} out={row['output_tokens']} cost=${row['cost_usd']:.4f}")

    print(f"\n  Середня вартість welcome:   ${sum(welcome_costs)/len(welcome_costs):.4f}")
    print(f"  Середня вартість follow_up: ${sum(followup_costs)/len(followup_costs):.4f}")
    print(f"\n  Лог збережено: {TOKENS_LOG}")


if __name__ == "__main__":
    main()
