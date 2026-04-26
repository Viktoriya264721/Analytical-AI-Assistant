from __future__ import annotations

import uuid
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from agent.llm import build_llm
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from supabase import Client

from agent.context import fetch_available_months
from agent.state import CFOState
from agent.system_prompt import build_system_prompt
from agent.tools import make_tools

_WELCOME_PROMPT_TEMPLATE = """\
/no_think
Доступні місяці: {available_months}.
Використай інструменти щоб знайти 2 найважливіші фінансові теми. Відповідь стисло.

Формат:
- Привітання (1 речення)
- 2 теми з цифрами (по 1-2 речення кожна)
- Після кожної: QUESTION: <питання одним реченням>\
"""

_FOLLOW_UP_PROMPT_TEMPLATE = """\
/no_think
Контекст розмови:
{recent_messages}

Доступні місяці: {available_months}.
Запропонуй 2 нові фінансові питання, які було б цікаво обговорити далі. Не повторюй вже обговорене.

Формат:
QUESTION: <питання одним реченням>
QUESTION: <питання одним реченням>\
"""


def _should_continue(state: CFOState) -> Literal["tools", "__end__"]:
    """Route to tool execution or end based on the last assistant message."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return END


def build_graph(supabase: Client, model_name: str) -> object:
    """Compile and return the CFO LangGraph agent.

    The graph follows a ReAct loop: the LLM decides which tools to call,
    the ``ToolNode`` executes them, and the LLM is invoked again with
    the results until no more tool calls are required.

    Args:
        supabase: Authenticated Supabase client injected into all tools.
        model_name: Model identifier passed to :func:`build_llm`.

    Returns:
        Compiled LangGraph ``CompiledGraph`` with an in-memory checkpointer.
    """
    tools = make_tools(supabase)
    llm = build_llm(model_name, temperature=0).bind_tools(tools)

    def agent_node(state: CFOState) -> dict:
        system_prompt = build_system_prompt(
            state["available_months"], state["latest_month"]
        )
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    workflow = StateGraph(CFOState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver(), debug=False)


def restore_session(
    graph: object,
    session_id: str,
    messages: list[dict[str, str]],
    available_months: list[str],
    latest_month: str,
) -> None:
    """Seed the graph checkpointer with messages loaded from Supabase.

    Call this once after loading an old conversation from the database so
    the LangGraph MemorySaver has the correct message history for the
    thread identified by *session_id*.

    Args:
        graph: Compiled LangGraph returned by :func:`build_graph`.
        session_id: Conversation identifier used as the LangGraph thread ID.
        messages: List of ``{"role": str, "content": str}`` dicts from Supabase.
        available_months: Months to inject into the graph state.
        latest_month: Most recent month to inject into the graph state.
    """
    lc_messages = []
    for m in messages:
        if m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    config = {"configurable": {"thread_id": session_id}}
    graph.update_state(
        config,
        {
            "messages": lc_messages,
            "available_months": available_months,
            "latest_month": latest_month,
        },
    )


def chat(
    graph: object,
    session_id: str,
    user_input: str,
    available_months: list[str],
    latest_month: str,
) -> str:
    """Send a user message and return the assistant reply.

    Args:
        graph: Compiled LangGraph returned by :func:`build_graph`.
        session_id: Conversation identifier used as the LangGraph thread ID.
        user_input: The user's plain-text message.
        available_months: Current list of months available in the database.
        latest_month: Most recent available month in YYYY-MM format.

    Returns:
        The assistant's final plain-text response after all tool calls complete.
    """
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
    state = {
        "messages": [HumanMessage(content=user_input)],
        "available_months": available_months,
        "latest_month": latest_month,
    }
    result = graph.invoke(state, config=config)
    return result["messages"][-1].content


def generate_welcome(
    graph: object,
    available_months: list[str],
    latest_month: str,
) -> tuple[str, list[str]]:
    """Generate an opening welcome message for a new conversation.

    Invokes the graph with a special prompt that asks the agent to surface
    the two most important financial topics worth discussing right now.
    Uses a temporary thread ID so this call does not pollute conversation state.

    Args:
        graph: Compiled LangGraph returned by :func:`build_graph`.
        available_months: Current list of months available in the database.
        latest_month: Most recent available month in YYYY-MM format.

    Returns:
        Tuple of (welcome message, list of up to 3 suggested questions extracted
        from QUESTION: markers in the message).
    """
    import re

    ephemeral_thread = f"welcome-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": ephemeral_thread}, "recursion_limit": 50}
    months_str = ", ".join(available_months) if available_months else latest_month
    prompt = _WELCOME_PROMPT_TEMPLATE.format(available_months=months_str)
    state = {
        "messages": [HumanMessage(content=prompt)],
        "available_months": available_months,
        "latest_month": latest_month,
    }
    result = graph.invoke(state, config=config)
    content = result["messages"][-1].content

    questions = re.findall(r"QUESTION:\s*(.+)", content)
    clean_content = re.sub(r"QUESTION:\s*.+", "", content).strip()

    return clean_content, questions[:2]


def generate_follow_up_questions(
    graph: object,
    chat_messages: list[dict],
    available_months: list[str],
    latest_month: str,
) -> list[str]:
    """Generate follow-up topic suggestions based on recent conversation context.

    Uses an ephemeral thread so this call does not affect the real conversation state.

    Args:
        graph: Compiled LangGraph returned by :func:`build_graph`.
        chat_messages: Full conversation history as list of {"role", "content"} dicts.
        available_months: Current list of months available in the database.
        latest_month: Most recent available month in YYYY-MM format.

    Returns:
        List of up to 2 suggested follow-up questions.
    """
    import re

    recent = chat_messages[-4:] if len(chat_messages) > 4 else chat_messages
    recent_text = "\n".join(
        f"{m['role'].upper()}: {m['content'][:200]}" for m in recent
    )

    ephemeral_thread = f"followup-{uuid.uuid4()}"
    config = {"configurable": {"thread_id": ephemeral_thread}, "recursion_limit": 50}
    months_str = ", ".join(available_months) if available_months else latest_month
    prompt = _FOLLOW_UP_PROMPT_TEMPLATE.format(
        recent_messages=recent_text,
        available_months=months_str,
    )
    state = {
        "messages": [HumanMessage(content=prompt)],
        "available_months": available_months,
        "latest_month": latest_month,
    }
    result = graph.invoke(state, config=config)
    content = result["messages"][-1].content

    questions = re.findall(r"QUESTION:\s*(.+)", content)
    return questions[:2]
