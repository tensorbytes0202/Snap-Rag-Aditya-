import json
import logging

from app.cache.redis_client import redis_call


_log = logging.getLogger(__name__)


def get_embedding(question: str):

    key = f"embedding:{question}"

    data = redis_call(lambda r: r.get(key), "computing the embedding instead")

    if data:

        _log.info("Embedding cache hit")

        return json.loads(data)

    _log.info("Embedding cache miss")

    return None


def set_embedding(question: str, embedding):

    key = f"embedding:{question}"

    redis_call(
        lambda r: r.set(key, json.dumps(embedding), ex=86400),
        "embedding not cached",
    )
