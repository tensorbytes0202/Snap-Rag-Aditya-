import logging

from sentence_transformers import CrossEncoder

from app.core.config import settings


_log = logging.getLogger(__name__)

_cross_encoder = None


def get_cross_encoder():

    global _cross_encoder

    if _cross_encoder is None:
        _log.info("Loading cross encoder model...")
        _cross_encoder = CrossEncoder(
            settings.CROSS_ENCODER_MODEL
        )

    return _cross_encoder