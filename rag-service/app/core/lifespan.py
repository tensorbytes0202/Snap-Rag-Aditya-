import logging
import os

from contextlib import asynccontextmanager

from app.api.dependencies.embedding import get_embedding_model
from app.api.dependencies.reranker import get_cross_encoder
from app.api.dependencies.qdrant import qdrant_client
from app.core.config import settings


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):

    logger.info("Starting RAG Service")

    # ------------------------------------------------------
    # Storage directories (must exist before anything tries
    # to read/write the registry, raw PDFs or BM25 indexes).
    # ------------------------------------------------------

    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    os.makedirs(settings.RAW_DATA_PATH, exist_ok=True)
    os.makedirs(settings.BM25_INDEX_DIR, exist_ok=True)

    # ------------------------------------------------------
    # Warm up dependencies. A failure here (e.g. Qdrant is
    # not reachable yet, or a model can't be downloaded) must
    # not prevent the service from starting -- it should just
    # be logged so the container comes up and later requests
    # can retry.
    # ------------------------------------------------------

    try:
        get_embedding_model()
        get_cross_encoder()
        qdrant_client.get_collections()
    except Exception:
        logger.warning("Startup warm-up failed", exc_info=True)

    logger.info("Startup Complete")

    yield

    logger.info("Shutting Down RAG Service")
