from __future__ import annotations

import json

from langchain_core.messages import HumanMessage
from supabase import Client

from agent.context import build_agent_context, resolve_names_in_text
from agent.llm import build_llm
from agent.prompt import build_full_prompt


def generate_monthly_report(
    supabase: Client,
    target_month: str,
    model_name: str | None = None,
) -> str:
    """Generate the structured monthly financial report for *target_month*.

    Fetches the full financial context from Supabase, assembles the prompt,
    and calls the LLM to produce a Markdown report.

    Args:
        supabase: Authenticated Supabase client.
        target_month: Month in YYYY-MM format, e.g. ``"2025-11"``.
        model_name: Model identifier passed to ``build_llm``; defaults to the
            value configured in the environment.

    Returns:
        Generated report as a Markdown string with real names resolved.
    """
    summary_dict = build_agent_context(supabase, target_month)
    summary_json = json.dumps(summary_dict, ensure_ascii=False, indent=2)
    full_prompt = build_full_prompt(summary_json)

    llm = build_llm(model_name, temperature=0, max_tokens=4096)
    response = llm.invoke([HumanMessage(content=full_prompt)])
    return resolve_names_in_text(response.content, supabase)
