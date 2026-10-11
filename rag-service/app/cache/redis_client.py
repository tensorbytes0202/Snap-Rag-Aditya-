"""One shared Redis client for the caches, with a small circuit breaker.

The caches are only an optimization. If Redis is down, a call fails once
(fast: no client-side retries, short timeouts) and Redis is then skipped
for COOLDOWN_SECONDS, so an outage costs one short delay instead of a
timeout on every request.
"""

import logging
import time

import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.core.config import settings


_log = logging.getLogger(__name__)

COOLDOWN_SECONDS = 30

redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=settings.REDIS_DB,
    decode_responses=True,
    socket_connect_timeout=1,
    socket_timeout=1,
    retry=Retry(NoBackoff(), 0),
)

_skip_until = 0.0


def redis_call(operation, on_failure: str):
    """Run `operation(redis_client)` and return its result, or None when Redis
    is unavailable (now, or within the cooldown after a recent failure)."""

    global _skip_until

    if time.monotonic() < _skip_until:
        return None

    try:
        return operation(redis_client)
    except redis.RedisError as exc:
        _skip_until = time.monotonic() + COOLDOWN_SECONDS
        _log.warning(
            "Redis unavailable (%s); %s. Skipping Redis for %ds.",
            exc, on_failure, COOLDOWN_SECONDS,
        )
        return None
