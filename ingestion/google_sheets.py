"""Google Sheets ingestion: load domain tables into raw DataFrames."""

import os
import pandas as pd
import gspread
from typing import Dict
from google.oauth2.service_account import Credentials


SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

SHEETS_MAP = {
    "daily_revenue":       "daily_revenue",
    "expenses":            "expenses",
    "amortization":        "amortization",
    "specialist_capacity": "specialist_capacity",
    "specialist_activity": "specialist_activity",
    "specialist_payouts":  "specialist_payouts",
}


def _get_client() -> gspread.Client:
    """Create an authenticated gspread client from a service-account file.

    Reads the path to the JSON credentials file from the
    ``GOOGLE_CREDENTIALS_PATH`` environment variable.

    Returns:
        Authorised :class:`gspread.Client` instance.

    Raises:
        RuntimeError: When ``GOOGLE_CREDENTIALS_PATH`` is not set.
    """
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")

    if not creds_path:
        raise RuntimeError("GOOGLE_CREDENTIALS_PATH is not set")

    creds = Credentials.from_service_account_file(
        creds_path,
        scopes=SCOPES,
    )
    return gspread.authorize(creds)


def load_raw_tables() -> Dict[str, pd.DataFrame]:
    """Load all domain worksheets from Google Sheets into DataFrames.

    Reads the spreadsheet ID from the ``GOOGLE_SHEET_ID`` environment
    variable and fetches each sheet listed in :data:`SHEETS_MAP`.
    Missing worksheets are returned as empty DataFrames.

    Returns:
        Mapping of canonical table name to raw :class:`pandas.DataFrame`.

    Raises:
        RuntimeError: When ``GOOGLE_SHEET_ID`` is not set.
    """
    spreadsheet_id = os.getenv("GOOGLE_SHEET_ID")

    if not spreadsheet_id:
        raise RuntimeError("GOOGLE_SHEET_ID is not set")

    client = _get_client()
    spreadsheet = client.open_by_key(spreadsheet_id)

    tables: Dict[str, pd.DataFrame] = {}

    for sheet_name, canonical_name in SHEETS_MAP.items():
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            records = worksheet.get_all_records()
            df = pd.DataFrame(records)

            df.columns = (
                df.columns
                .map(lambda x: str(x).strip().replace("\n", " "))
            )

            tables[canonical_name] = df
            print(f"Loaded sheet: {sheet_name} ({len(df)} rows)")

        except gspread.WorksheetNotFound:
            print(f"Sheet not found: {sheet_name}")
            tables[canonical_name] = pd.DataFrame()

    return tables
