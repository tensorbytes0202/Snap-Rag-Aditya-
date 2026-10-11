import logging

from sentence_transformers import SentenceTransformer

from app.core.config import settings


_log = logging.getLogger(__name__)

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:
        _log.info("Loading embedding model...")
        _embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL
        )

    return _embedding_model