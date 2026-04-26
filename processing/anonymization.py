import pandas as pd
from typing import Dict
from supabase import Client

_REHAB_TABLES = {"specialist_activity", "specialist_capacity"}


def _infer_person_type(name: str, tables: Dict[str, pd.DataFrame]) -> str:
    """Infer whether a person is rehab staff or a support employee.

    A person is classified as ``"rehab"`` if they appear in any rehab table;
    otherwise they are classified as ``"employee"``.

    Args:
        name: Real name to look up.
        tables: Mapping of table name to DataFrame.

    Returns:
        ``"rehab"`` or ``"employee"``.
    """
    for t in _REHAB_TABLES:
        df = tables.get(t, pd.DataFrame())
        if not df.empty and "person" in df.columns:
            if name in df["person"].values:
                return "rehab"

    return "employee"


def _next_anonymous_id(supabase: Client, person_type: str) -> str:
    """Generate the next sequential anonymous ID for a given person type.

    ID format:
        - ``rehab``    → ``rehab_01``, ``rehab_02``, …
        - ``employee`` → ``empl_01``,  ``empl_02``,  …
        - ``owner``    → ``owner_1``,  ``owner_2``,  …

    Args:
        supabase: Authenticated Supabase client.
        person_type: One of ``"rehab"``, ``"employee"``, or ``"owner"``.

    Returns:
        New unique anonymous ID string.
    """
    prefix_map = {"rehab": "rehab_", "employee": "empl_", "owner": "owner_"}
    prefix = prefix_map[person_type]

    response = (
        supabase.table("persons")
        .select("anonymous_id")
        .like("anonymous_id", f"{prefix}%")
        .execute()
    )

    nums = []
    for row in (response.data or []):
        try:
            nums.append(int(row["anonymous_id"].replace(prefix, "")))
        except ValueError:
            pass

    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}{next_num:02d}"


def _build_name_map(
    supabase: Client,
    normalized_tables: Dict[str, pd.DataFrame],
) -> Dict[str, str]:
    """Build a ``real_name → anonymous_id`` mapping for all persons in the data.

    New names are registered in the ``persons`` table before being added to
    the mapping.

    Args:
        supabase: Authenticated Supabase client.
        normalized_tables: Cleaned and normalised domain tables.

    Returns:
        Mapping of real name to anonymous ID.
    """
    all_names: set = set()
    for df in normalized_tables.values():
        if df.empty or "person" not in df.columns:
            continue
        all_names.update(df["person"].dropna().unique())

    if not all_names:
        return {}

    response = supabase.table("persons").select("real_name, anonymous_id").execute()
    known: Dict[str, str] = {
        row["real_name"]: row["anonymous_id"] for row in (response.data or [])
    }

    name_map = dict(known)

    for name in sorted(all_names):
        if name in known:
            continue

        person_type = _infer_person_type(name, normalized_tables)
        anonymous_id = _next_anonymous_id(supabase, person_type)

        supabase.table("persons").insert({
            "real_name":    name,
            "anonymous_id": anonymous_id,
            "person_type":  person_type,
        }).execute()

        name_map[name] = anonymous_id
        known[name] = anonymous_id
        print(f"  New person: '{name}' -> '{anonymous_id}' ({person_type})")

    return name_map


def anonymize_tables(
    normalized_tables: Dict[str, pd.DataFrame],
    supabase: Client,
) -> Dict[str, pd.DataFrame]:
    """Replace real names with anonymous IDs in all tables.

    New persons encountered in the data are automatically registered in the
    ``persons`` table before substitution.

    Args:
        normalized_tables: Cleaned and normalised domain tables.
        supabase: Authenticated Supabase client.

    Returns:
        Tables with ``person`` columns replaced by anonymous IDs.
    """
    name_map = _build_name_map(supabase, normalized_tables)

    if not name_map:
        return normalized_tables

    anonymized = {}
    for table_name, df in normalized_tables.items():
        if df.empty or "person" not in df.columns:
            anonymized[table_name] = df
            continue

        df = df.copy()
        df["person"] = df["person"].apply(
            lambda x: name_map.get(x, x) if not pd.isna(x) else pd.NA
        )
        anonymized[table_name] = df

    return anonymized
