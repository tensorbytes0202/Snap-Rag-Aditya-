import logging

from app.core.config import settings


logging.basicConfig(
    level=settings.LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# httpx logs every request at INFO; keep one line per call.
logging.getLogger("httpx").setLevel(logging.WARNING)


logger = logging.getLogger("rag-service")
