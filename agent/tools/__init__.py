from __future__ import annotations

from supabase import Client

from agent.tools.analysis import make_analysis_tools
from agent.tools.financial import make_financial_tools
from agent.tools.operations import make_operations_tools
from agent.tools.people import make_people_tools
from agent.tools.persons import make_persons_tools


def make_tools(supabase: Client) -> list:
    """Collect all CFO agent tools with an injected Supabase client.

    Args:
        supabase: Authenticated Supabase client shared across all tool groups.

    Returns:
        Flat list of LangChain tool callables ready to bind to an LLM.
    """
    return (
        make_persons_tools(supabase)
        + make_financial_tools(supabase)
        + make_people_tools(supabase)
        + make_operations_tools(supabase)
        + make_analysis_tools(supabase)
    )
