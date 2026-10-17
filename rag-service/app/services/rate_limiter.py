"""Demo quotas for guest users.

Guests (user ids starting with "guest-", minted by POST /auth/guest) may ask
GUEST_QUERY_LIMIT questions and upload GUEST_UPLOAD_LIMIT PDFs, and each
visitor (IP) may start GUEST_SESSIONS_PER_IP guest sessions a day. Every
other user is unlimited.

Quota is reserved before the expensive work (consume) and given back if that
work fails (refund), so concurrent requests can't overshoot the limit and an
outage doesn't cost a reviewer a question. Counters live in storage/usage.json
so a restart doesn't reset them.
"""

import json
import os
import threading
import time
from collections import deque
from datetime import date

from app.core.config import settings


GUEST_PREFIX = "guest-"

_lock = threading.Lock()


class QuotaExceeded(Exception):
    """A guest has used up a demo quota."""


def is_guest(user_id: str) -> bool:

    return user_id.startswith(GUEST_PREFIX)


def _limit(kind: str) -> int:

    return {"queries": settings.GUEST_QUERY_LIMIT, "uploads": settings.GUEST_UPLOAD_LIMIT}[kind]


def _load() -> dict:

    if not os.path.isfile(settings.USAGE_FILE):
        return {"users": {}, "ip_sessions": {}}

    with open(settings.USAGE_FILE, encoding="utf-8") as f:
        return json.load(f)


def _save(data: dict) -> None:

    os.makedirs(os.path.dirname(settings.USAGE_FILE), exist_ok=True)

    tmp_path = settings.USAGE_FILE + ".tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    os.replace(tmp_path, settings.USAGE_FILE)


def consume(user_id: str, kind: str) -> None:
    """Reserve one `kind` ("queries" or "uploads"); raises QuotaExceeded
    when a guest has none left. No-op for other users."""

    if not is_guest(user_id):
        return

    with _lock:

        data = _load()
        used = data["users"].setdefault(user_id, {}).get(kind, 0)

        if used >= _limit(kind):
            noun = "questions" if kind == "queries" else "PDF uploads"
            raise QuotaExceeded(f"Demo limit reached: you have used all {_limit(kind)} {noun}.")

        data["users"][user_id][kind] = used + 1
        _save(data)


def refund(user_id: str, kind: str) -> None:
    """Give back a reservation whose work failed."""

    if not is_guest(user_id):
        return

    with _lock:

        data = _load()
        counts = data["users"].get(user_id, {})

        if counts.get(kind, 0) > 0:
            counts[kind] -= 1
            _save(data)


def usage(user_id: str) -> dict:

    if not is_guest(user_id):
        return {"guest": False}

    with _lock:
        counts = _load()["users"].get(user_id, {})

    return {
        "guest": True,
        "queries_used": counts.get("queries", 0),
        "query_limit": settings.GUEST_QUERY_LIMIT,
        "uploads_used": counts.get("uploads", 0),
        "upload_limit": settings.GUEST_UPLOAD_LIMIT,
        "max_pages": settings.GUEST_MAX_PAGES,
    }


def start_guest_session(ip: str) -> bool:
    """Count a new guest session for `ip`; False once today's cap is hit."""

    today = date.today().isoformat()
    key = f"{today}|{ip}"

    with _lock:

        data = _load()

        # Keep only today's counters.
        sessions = {k: v for k, v in data["ip_sessions"].items() if k.startswith(today + "|")}

        if sessions.get(key, 0) >= settings.GUEST_SESSIONS_PER_IP:
            return False

        sessions[key] = sessions.get(key, 0) + 1
        data["ip_sessions"] = sessions
        _save(data)

    return True


# ==========================================================
# Request rate (per user, per minute)
# ==========================================================
#
# Separate from the guest quotas above: this limits how fast any user can
# start questions (each one costs several LLM calls), not how many a guest
# has in total. Kept in memory; a restart simply starts the windows over.

_recent: dict[str, deque] = {}
_recent_lock = threading.Lock()

WINDOW_SECONDS = 60


def allow_request(user_id: str, now: float | None = None) -> bool:
    """Count a request for `user_id`; False when QUERIES_PER_MINUTE requests
    were already made in the last minute (0 turns the limit off)."""

    limit = settings.QUERIES_PER_MINUTE

    if limit <= 0:
        return True

    now = time.monotonic() if now is None else now

    with _recent_lock:

        window = _recent.setdefault(user_id, deque())

        while window and now - window[0] >= WINDOW_SECONDS:
            window.popleft()

        if len(window) >= limit:
            return False

        window.append(now)

    return True
