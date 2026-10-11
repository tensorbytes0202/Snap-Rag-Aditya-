import logging
import os

from contextlib import asynccontextmanager

from app.api.dependencies.embedding import get_embedding_model
from app.api.dependencies.reranker import get_cross_encoder
from app.api.dependencies.qdrant import qdrant_client
from app.core.config import settings
from app.services.cleanup_service import sweep_stale_temp_files


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
    # Remove temp files a killed/crashed process left behind (a
    # partially written upload or BM25 index that never got cleaned
    # up). Only files older than STALE_TEMP_FILE_AGE_SECONDS are
    # touched, so an upload genuinely in flight right now is safe.
    # ------------------------------------------------------

    sweep_stale_temp_files()

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
