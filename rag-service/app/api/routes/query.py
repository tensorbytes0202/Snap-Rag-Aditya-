import json
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.exceptions import LLMProviderError
from app.core.security import get_current_user_id
from app.services.query_service import process_query
from app.services.query_graph import stream_query
from app.services.document_registry import get_active_document


logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):

    question: str

    documents: List[str] = []


@router.post("/query")
def query_document(
    request: QueryRequest,
    user_id: str = Depends(get_current_user_id),
):

    active_document = get_active_document(user_id)

    logger.debug(
        "Query received: question=%r active_document=%r",
        request.question,
        active_document,
    )

    try:
        return process_query(
            request.question,
            active_document,
        )
    except LLMProviderError:
        logger.exception("Language model request failed")
        raise HTTPException(
            status_code=502,
            detail="Language model request failed",
        )


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event) + "\n\n"


def _sse_events(question: str, active_document: str | None):

    try:
        for event in stream_query(question, active_document):
            yield _sse(event)
    except LLMProviderError:
        logger.exception("Language model request failed")
        yield _sse({"type": "error", "message": "Language model request failed"})
    except Exception:
        # Without this, an unexpected error kills the stream mid-response and
        # the client only sees a dropped connection.
        logger.exception("Streaming query failed")
        yield _sse({"type": "error", "message": "Something went wrong while answering. Check the backend log."})

    yield _sse({"type": "done"})


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    user_id: str = Depends(get_current_user_id),
):

    return StreamingResponse(
        _sse_events(request.question, get_active_document(user_id)),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
