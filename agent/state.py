from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class CFOState(TypedDict):
    """State for the CFO agent graph.

    Attributes:
        messages: Conversation history managed by LangGraph.
            New messages are appended via the add_messages reducer.
        available_months: Sorted list of months present in the database,
            formatted as YYYY-MM. Populated once at graph initialisation
            and injected into the system prompt.
        latest_month: The most recent month in available_months.
    """

    messages: Annotated[list, add_messages]
    available_months: list[str]
    latest_month: str
