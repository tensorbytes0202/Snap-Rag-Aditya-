"""Service health and metrics (no authentication: for load balancers and scrapers)."""

import os

from fastapi import APIRouter, Response

from app.cache.redis_client import redis_call
from app.core import metrics
from app.core.config import settings


router = APIRouter(tags=["system"])


def _qdrant() -> dict:

    from app.api.dependencies.qdrant import qdrant_client

    try:
        qdrant_client.get_collections()
        return {"ok": True}
    except Exception:
        return {"ok": False}


def _redis() -> dict:

    # The answer cache is optional, so a down Redis degrades the service
    # instead of failing it.
    ok = redis_call(lambda r: r.ping(), "health check") is True

    return {"ok": ok, "required": False}


def _storage() -> dict:

    return {"ok": os.path.isdir(settings.STORAGE_DIR) and os.access(settings.STORAGE_DIR, os.W_OK)}


@router.get("/healthz")
def healthz(response: Response):
    """Dependency status. 503 when a required dependency (Qdrant, storage,
    LLM configuration) is down; Redis only degrades."""

    checks = {
        "qdrant": _qdrant(),
        "redis": _redis(),
        "storage": _storage(),
        "llm": {"ok": bool(settings.LLM_GATEWAY_TOKEN and settings.LLM_GATEWAY_URL)},
    }

    healthy = all(check["ok"] for name, check in checks.items() if check.get("required", True))

    if not healthy:
        response.status_code = 503

    return {"status": "ok" if healthy else "unavailable", "checks": checks}


@router.get("/metrics")
def prometheus_metrics():

    return Response(metrics.render_prometheus(), media_type="text/plain; version=0.0.4")
