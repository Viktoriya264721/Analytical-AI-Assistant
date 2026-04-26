from __future__ import annotations

import json

from langchain_core.tools import tool
from supabase import Client


def make_persons_tools(supabase: Client) -> list:
    """Create person lookup tool with an injected Supabase client."""

    @tool
    def find_person(name: str) -> str:
        """Find a person's anonymous ID by their real name or part of their name.

        Use this tool before any query that involves a specific person by name
        (e.g. salary, efficiency, services for "Ivan" or "Maria").
        Returns anonymous_id and person_type — never exposes the real name.

        Args:
            name: Full or partial real name to search for, e.g. "іван" or "марія".

        Returns:
            JSON string. On success: {"found": true, "matches": [{"anonymous_id": ..., "person_type": ...}]}.
            On failure: {"found": false, "message": ...}.
        """
        response = (
            supabase.table("persons")
            .select("anonymous_id, person_type")
            .ilike("real_name", f"%{name}%")
            .execute()
        )

        matches = response.data or []

        if not matches:
            return json.dumps(
                {"found": False, "message": f"No person found matching '{name}'"},
                ensure_ascii=False,
            )

        return json.dumps({"found": True, "matches": matches}, ensure_ascii=False)

    @tool
    def resolve_name(anonymous_id: str) -> str:
        """Resolve an anonymous ID back to a real person's name for display purposes only.

        Use this tool ONLY at the end of a response to show the user a human-readable name.
        Never use the returned name as input to other queries — always use anonymous_id for data lookups.

        Args:
            anonymous_id: The anonymous identifier, e.g. "EMP-042".

        Returns:
            JSON string. On success: {"found": true, "real_name": "..."}.
            On failure: {"found": false, "message": ...}.
        """
        response = (
            supabase.table("persons")
            .select("real_name")
            .eq("anonymous_id", anonymous_id)
            .execute()
        )

        data = response.data or []

        if not data:
            return json.dumps(
                {"found": False, "message": f"No person found with anonymous_id '{anonymous_id}'"},
                ensure_ascii=False,
            )

        return json.dumps({"found": True, "real_name": data[0]["real_name"]}, ensure_ascii=False)

    return [find_person, resolve_name]
