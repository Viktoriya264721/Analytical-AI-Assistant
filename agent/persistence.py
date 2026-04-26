from __future__ import annotations

import uuid
from typing import Any

from supabase import Client

_TABLE = "chat_messages"


def new_session_id() -> str:
    """Generate a unique conversation session identifier."""
    return str(uuid.uuid4())


def save_message(
    supabase: Client,
    session_id: str,
    role: str,
    content: str,
    username: str = "",
) -> None:
    """Persist a single chat message to Supabase.

    Args:
        supabase: Authenticated Supabase client.
        session_id: UUID identifying the conversation session.
        role: Message author – ``"user"`` or ``"assistant"``.
        content: Plain-text message body.
        username: The authenticated user who owns this conversation.
    """
    supabase.table(_TABLE).insert(
        {"session_id": session_id, "role": role, "content": content, "username": username}
    ).execute()


def load_conversation(supabase: Client, session_id: str) -> list[dict[str, str]]:
    """Load all messages for a given session, ordered chronologically.

    Args:
        supabase: Authenticated Supabase client.
        session_id: UUID identifying the conversation session.

    Returns:
        List of ``{"role": str, "content": str}`` dicts.
    """
    response = (
        supabase.table(_TABLE)
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .execute()
    )
    if not response.data:
        return []
    return [{"role": r["role"], "content": r["content"]} for r in response.data]


def list_conversations(supabase: Client, username: str = "") -> list[dict[str, Any]]:
    """Return a summary list of past conversations for the sidebar.

    Each entry contains ``session_id``, ``preview`` (first user message
    truncated to 60 chars, falling back to the first assistant message),
    and ``created_at``.

    Args:
        supabase: Authenticated Supabase client.

    Returns:
        List of conversation summary dicts, sorted newest-first.
    """
    query = supabase.table(_TABLE).select("session_id, role, content, created_at")
    if username:
        query = query.eq("username", username)
    response = query.order("created_at", desc=False).execute()
    if not response.data:
        return []

    sessions: dict[str, dict[str, Any]] = {}

    for row in response.data:
        sid = row["session_id"]
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "preview": "",
                "created_at": row["created_at"],
                "_has_user": False,
            }

        entry = sessions[sid]

        if row["role"] == "user" and not entry["_has_user"]:
            text = row["content"]
            entry["preview"] = text[:60] + ("..." if len(text) > 60 else "")
            entry["_has_user"] = True
        elif row["role"] == "assistant" and not entry["preview"]:
            text = row["content"].replace("\n", " ").strip()
            entry["preview"] = text[:50] + ("..." if len(text) > 50 else "")

    result = sorted(sessions.values(), key=lambda x: x["created_at"], reverse=True)
    for item in result:
        item.pop("_has_user", None)
    return result
