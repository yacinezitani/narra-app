"""
Rate limiting + owner bypass.

Protects the (free, single-instance) TTS backend from abuse without requiring
accounts: each client IP gets a sliding-window budget of synthesis requests.
You — the owner — bypass everything by sending a secret key.

How "unlimited for me" works:
  * Set the env var ``OWNER_KEY`` on the backend to a long random secret.
  * The frontend sends it as the ``X-Narra-Key`` header when present (it reads
    the key from localStorage, so only the browser where *you* stored it gets
    unlimited access — visitors never see it).
  * Requests with the matching key skip the rate limit (and the usage quota).

Config (env):
  OWNER_KEY            secret string; requests with X-Narra-Key == this are owner
  RATE_LIMIT_PER_MIN   per-IP synthesis requests / minute   (default 8;   0 = off)
  RATE_LIMIT_PER_DAY   per-IP synthesis requests / day      (default 200; 0 = off)

Note: the store is in-memory, so limits are per server instance. That's fine for
one free HF Space; a multi-instance / paid setup would move this to Redis.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

log = logging.getLogger("kokoro")

OWNER_KEY = os.getenv("OWNER_KEY", "")
PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "8"))
PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "200"))

_MINUTE = 60
_DAY = 86_400

# client ip -> deque of request timestamps (kept for the last 24h)
_hits: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def is_owner(x_narra_key: str | None) -> bool:
    """True when the request carries the owner key (and one is configured)."""
    return bool(OWNER_KEY) and x_narra_key == OWNER_KEY


def _client_ip(request: Request) -> str:
    """Best-effort client IP, honoring the proxy header HF Spaces sits behind."""
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        # First hop is the original client.
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request, owner: bool) -> None:
    """Charge one request against the caller's IP budget, or raise 429.

    No-op for the owner and when both limits are disabled (set to 0).
    """
    if owner or (PER_MIN <= 0 and PER_DAY <= 0):
        return

    ip = _client_ip(request)
    now = time.time()

    with _lock:
        q = _hits[ip]
        # Drop timestamps older than a day so the window slides and memory stays bounded.
        cutoff = now - _DAY
        while q and q[0] < cutoff:
            q.popleft()

        if PER_DAY > 0 and len(q) >= PER_DAY:
            retry = int(q[0] + _DAY - now) + 1
            raise _too_many(retry, "daily")

        if PER_MIN > 0:
            minute_ago = now - _MINUTE
            recent = sum(1 for t in q if t >= minute_ago)
            if recent >= PER_MIN:
                # Oldest hit within this minute determines when a slot frees up.
                oldest_in_min = next(t for t in q if t >= minute_ago)
                retry = int(oldest_in_min + _MINUTE - now) + 1
                raise _too_many(retry, "per-minute")

        q.append(now)

        # Opportunistic cleanup so idle IPs don't linger forever.
        if len(_hits) > 10_000:
            for k in [k for k, v in _hits.items() if not v or v[-1] < cutoff]:
                _hits.pop(k, None)


def _too_many(retry_after: int, kind: str) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail={
            "error": "rate_limited",
            "message": f"Rate limit reached ({kind}). Try again in {retry_after}s.",
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )
