import os
from supabase import create_client
from typing import Optional


def get_supabase() -> Optional[object]:
    """Initialise a Supabase client from environment variables.

    Returns:
        Authenticated Supabase client, or ``None`` when credentials are
        not set (dry-run mode).
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not key:
        return None

    return create_client(url, key)


supabase = get_supabase()
