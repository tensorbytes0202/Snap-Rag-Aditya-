import json
import hashlib
import logging

from app.cache.redis_client import redis_call
from app.core.config import settings


_log = logging.getLogger(__name__)

# Namespaces every answer-cache key so it can be told apart from other
# keys sharing this Redis DB (e.g. embedding_cache's "embedding:*") and
# safely bulk-cleared without touching them.
CACHE_KEY_PREFIX = "answer_cache:"


def make_cache_key(question: str):

    return CACHE_KEY_PREFIX + hashlib.sha256(

        question.lower().strip().encode()

    ).hexdigest()


# The answer cache is an optimization: if Redis is down, every call below
# degrades to "no cache" (see redis_client.redis_call) instead of failing.

def get_cache(question: str):

    key = make_cache_key(question)

    data = redis_call(lambda r: r.get(key), "skipping the answer cache")

    if data:

        _log.info("Cache hit")

        return json.loads(data)

    _log.info("Cache miss")

    return None


def set_cache(question: str, response):

    key = make_cache_key(question)

    redis_call(
        lambda r: r.setex(key, settings.CACHE_TTL, json.dumps(response)),
        "response not cached",
    )


def clear_all_cache():
    """
    Evict every cached answer.

    Cache keys are opaque sha256 hashes of "<document>::<question>" with
    no secondary index from document name to the keys that reference
    it, so a single document's entries can't be targeted directly.
    Rather than leave a stale answer reachable after a document is
    re-indexed, this clears the whole answer-cache namespace (still
    scoped to CACHE_KEY_PREFIX, so it never touches unrelated keys such
    as embedding_cache's, which shares this Redis DB).

    Called unconditionally on every re-upload, so a Redis outage must
    never fail the upload: redis_call logs the failure instead of raising.
    """

    def _clear(r):
        for key in r.scan_iter(match=f"{CACHE_KEY_PREFIX}*"):
            r.delete(key)

    redis_call(_clear, "cached answers not cleared")
