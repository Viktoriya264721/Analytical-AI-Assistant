import os
from dotenv import load_dotenv

env_path = os.getenv("ENV_PATH")
if not env_path:
    raise RuntimeError("ENV_PATH is not set")

load_dotenv(env_path)

from ingestion.google_sheets import load_raw_tables
from processing.cleaning import clean_tables
from processing.normalization import normalize_tables
from processing.validation import validate_all
from processing.anonymization import anonymize_tables
from db.attach_identity import attach_identity
from storage.db_writer import upsert_all
from analytics.compute import compute_for_updated_months
from config.supabase import supabase


def run_pipeline():
    """Run the full data ingestion and analytics pipeline.

    Loads raw data from Google Sheets, cleans, normalises, validates,
    attaches row identity hashes, anonymises person names, writes to
    Supabase, and recomputes monthly metrics for any affected months.

    Returns:
        dict[str, pd.DataFrame]: Anonymised tables after processing.
            In dry-run mode (no Supabase connection) returns identified
            but non-anonymised tables without writing to the database.
    """
    raw_tables = load_raw_tables()
    cleaned_tables = clean_tables(raw_tables)
    normalized_tables = normalize_tables(cleaned_tables)

    validate_all(normalized_tables)

    identified_tables = attach_identity(normalized_tables)

    if supabase is None:
        print("Supabase disabled (DRY MODE). Skipping anonymization and DB write.")
        return identified_tables

    anonymized_tables = anonymize_tables(identified_tables, supabase)

    print("\nSync: writing to database")
    print("-" * 50)

    touched_months = upsert_all(
        supabase=supabase,
        tables=anonymized_tables,
    )

    print("-" * 50)

    if touched_months:
        compute_for_updated_months(supabase, touched_months)

    print("Done.\n")
    return anonymized_tables


if __name__ == "__main__":
    run_pipeline()
