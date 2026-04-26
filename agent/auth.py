from __future__ import annotations

import bcrypt
from supabase import Client


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


def login_user(supabase: Client, username: str, password: str) -> tuple[bool, str, str | None]:
    """Verify credentials. Returns (success, message, role)."""
    result = (
        supabase.table("users")
        .select("hashed_password, role")
        .eq("username", username)
        .execute()
    )
    if not result.data:
        return False, "Невірний логін або пароль.", None

    row = result.data[0]
    if not verify_password(password, row["hashed_password"]):
        return False, "Невірний логін або пароль.", None

    return True, "Успішний вхід.", row.get("role", "user")


def create_user(supabase: Client, username: str, password: str, role: str = "user") -> tuple[bool, str]:
    """Admin creates a new user. Returns (success, message)."""
    if len(password) < 6:
        return False, "Пароль має бути не менше 6 символів."

    existing = (
        supabase.table("users")
        .select("id")
        .eq("username", username)
        .execute()
    )
    if existing.data:
        return False, "Користувач з таким логіном вже існує."

    supabase.table("users").insert({
        "username": username,
        "hashed_password": hash_password(password),
        "role": role,
    }).execute()
    return True, f"Користувача «{username}» створено."


def delete_user(supabase: Client, username: str) -> tuple[bool, str]:
    """Admin deletes a user. Returns (success, message)."""
    supabase.table("users").delete().eq("username", username).execute()
    return True, f"Користувача «{username}» видалено."


def list_users(supabase: Client) -> list[dict]:
    """Return all users (without passwords)."""
    result = supabase.table("users").select("username, role, created_at").order("created_at").execute()
    return result.data or []
