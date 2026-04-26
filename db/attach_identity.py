"""Row identity attachment: source_uid and content_hash computation."""

import hashlib
import pandas as pd
from typing import Dict

from db.event_identity import generate_source_uid, generate_content_hash


def _make_uid_unique(base_uid: str, seq: int) -> str:
    """Re-hash base_uid with a ``|seq`` suffix to deduplicate; returns base_uid unchanged when seq == 0."""
    if seq == 0:
        return base_uid
    raw = f"{base_uid}|{seq}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def attach_identity(
    normalized_tables: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Add ``source_uid`` and ``content_hash`` columns to every table.

    ``source_uid`` is a stable identifier derived from the row's natural key
    fields. Duplicate natural keys within the same batch are disambiguated by
    appending a sequence number before hashing.

    Args:
        normalized_tables: Cleaned and normalised domain tables.

    Returns:
        Same tables with two additional columns: ``source_uid`` and
        ``content_hash``.
    """
    result = {}

    for table_name, df in normalized_tables.items():
        if df.empty:
            result[table_name] = df
            continue

        tmp = df.copy()

        tmp["_base_uid"] = tmp.apply(
            lambda row: generate_source_uid(row, table_name), axis=1
        )
        tmp["_seq"] = tmp.groupby("_base_uid").cumcount()

        tmp["source_uid"] = tmp.apply(
            lambda row: _make_uid_unique(row["_base_uid"], row["_seq"]),
            axis=1,
        )
        tmp["content_hash"] = tmp.apply(
            lambda row: generate_content_hash(row, table_name), axis=1
        )

        tmp = tmp.drop(columns=["_base_uid", "_seq"])
        result[table_name] = tmp

    return result
