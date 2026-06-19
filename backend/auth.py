"""
Authentication + usage quota, backed by Supabase.

Two responsibilities:

  1. **Identify the user** from the Supabase access token the frontend sends as
     ``Authorization: Bearer <jwt>``. We verify the JWT locally with the
     project's JWT secret (fast, no network round-trip per request).
  2. **Meter + enforce quota** by calling the ``consume_quota`` SQL function with
     the service-role key, which atomically charges usage and returns whether the
     request is allowed under the user's plan cap.

Everything here is **opt-in**: if ``SUPABASE_URL`` is not configured, auth is
disabled and the app runs in open local-dev mode (no accounts, no limits) exactly
like the original Narra. That keeps local development friction-free while the
hosted SaaS turns it on via env vars.
"""

from __future__ import annotations

import logging
import os

import httpx
import jwt
from fastapi import Depends, Header, HTTPException

log = logging.getLogger("kokoro")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# Auth is active only when fully configured. Missing config => open local mode.
AUTH_ENABLED = bool(SUPABASE_URL and SUPABASE_JWT_SECRET and SUPABASE_SERVICE_ROLE_KEY)

if AUTH_ENABLED:
    log.info("Auth + quota ENABLED (Supabase).")
else:
    log.info("Auth + quota DISABLED (open local mode — set SUPABASE_* env to enable).")


def _verify_jwt(token: str) -> str:
    """Return the user id (``sub``) from a valid Supabase access token, or raise."""
    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            # Supabase tokens carry aud="authenticated".
            audience="authenticated",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(401, f"Invalid token: {exc}") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(401, "Token missing subject")
    return user_id


async def get_user_id(authorization: str | None = Header(default=None)) -> str | None:
    """FastAPI dependency: resolve the caller's user id.

    Returns ``None`` in open local mode (auth disabled). Otherwise requires a
    valid bearer token and returns the user id.
    """
    if not AUTH_ENABLED:
        return None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing bearer token")
    return _verify_jwt(authorization.split(" ", 1)[1].strip())


class QuotaExceeded(HTTPException):
    """402 Payment Required — the user hit their plan cap; prompt an upgrade."""

    def __init__(self, used: int, limit: int, plan: str):
        super().__init__(
            status_code=402,
            detail={
                "error": "quota_exceeded",
                "message": "You've reached your monthly limit. Upgrade to Pro for more.",
                "plan": plan,
                "chars_used": used,
                "char_limit": limit,
            },
        )


async def enforce_quota(user_id: str | None, chars: int) -> None:
    """Charge ``chars`` against the user's quota; raise :class:`QuotaExceeded` if over.

    No-op in open local mode. Calls the ``consume_quota`` Postgres function via
    PostgREST RPC using the service-role key so metering can't be bypassed by the
    client. The DB function is the single source of truth for plan caps.
    """
    if not AUTH_ENABLED or user_id is None:
        return

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/consume_quota",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
            },
            json={"p_user": user_id, "p_chars": chars},
        )
    if resp.status_code >= 400:
        # Fail closed: if metering is broken we don't want to give away free GPU.
        log.error("consume_quota RPC failed: %s %s", resp.status_code, resp.text)
        raise HTTPException(503, "Usage metering unavailable, please retry")

    # The function returns a single row: [{allowed, plan, chars_used, char_limit}]
    row = resp.json()[0]
    if not row["allowed"]:
        raise QuotaExceeded(row["chars_used"], row["char_limit"], row["plan"])
