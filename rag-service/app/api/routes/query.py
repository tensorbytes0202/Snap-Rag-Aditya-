import json
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.security import get_current_user_id
from app.services.query_service import process_query
from app.services.query_graph import stream_query
from app.services.document_registry import get_active_document
from app.services.session_store import append_turn, chat_exists, get_turns
from app.services.snapshot_engine import after_turn


logger = logging.getLogger(__name__)

router = APIRouter()


class QueryRequest(BaseModel):

    question: str

    documents: List[str] = []

    # Optional chat (from POST /chats). With it, recent turns are shown to
    # the LLM and this turn is recorded; without it nothing is stored.
    chat_id: str | None = None


def _recent_history(user_id: str, chat_id: str | None) -> list[dict]:
    """The caller's last HISTORY_TURNS turns ([] without a chat)."""

    if chat_id is None:
        return []

    if not chat_exists(user_id, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")

    return get_turns(user_id, chat_id)[-settings.HISTORY_TURNS:]


@router.post("/query")
def query_document(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):

    history = _recent_history(user_id, request.chat_id)

    active_document = get_active_document(user_id)

    logger.debug(
        "Query received: question=%r active_document=%r",
        request.question,
        active_document,
    )

    try:
        response = process_query(
            request.question,
            active_document,
            history,
            (user_id, request.chat_id) if request.chat_id else None,
        )
    except LLMProviderError:
        logger.exception("Language model request failed")
        raise HTTPException(
            status_code=502,
            detail="Language model request failed",
        )

    if request.chat_id:

        verification = response.get("verification")

        append_turn(
            user_id,
            request.chat_id,
            request.question,
            response["answer"],
            document=active_document,
            citations=response.get("citations", []),
            verified=verification.get("supported", False) if isinstance(verification, dict) else False,
            confidence=response.get("confidence"),
        )

        # Context check + snapshot run after the answer is sent.
        background_tasks.add_task(after_turn, user_id, request.chat_id)

    return response


def _sse(event: dict) -> str:
    return "data: " + json.dumps(event) + "\n\n"


def _sse_events(question: str, active_document: str | None, history: list, session=None, record=None):
    """`record(answer, citations)` is called once a complete answer has
    streamed, to store the turn; it is skipped when the stream ends in an
    error."""

    tokens, citations, failed = [], [], False

    try:
        for event in stream_query(question, active_document, history, session):
            if event["type"] == "token":
                tokens.append(event["token"])
            elif event["type"] == "citations":
                citations = event["citations"]
            elif event["type"] == "error":
                failed = True
            yield _sse(event)
    except LLMProviderError:
        failed = True
        logger.exception("Language model request failed")
        yield _sse({"type": "error", "message": "Language model request failed"})
    except Exception:
        failed = True
        # Without this, an unexpected error kills the stream mid-response and
        # the client only sees a dropped connection.
        logger.exception("Streaming query failed")
        yield _sse({"type": "error", "message": "Something went wrong while answering. Check the backend log."})

    if record and tokens and not failed:
        record("".join(tokens), citations)

    yield _sse({"type": "done"})


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):

    history = _recent_history(user_id, request.chat_id)

    active_document = get_active_document(user_id)

    def record(answer: str, citations: list):
        append_turn(
            user_id, request.chat_id, request.question, answer,
            document=active_document, citations=citations,
        )

    if request.chat_id:
        # Runs after the stream ends; skips itself if no turn was recorded.
        background_tasks.add_task(after_turn, user_id, request.chat_id)

    return StreamingResponse(
        _sse_events(
            request.question,
            active_document,
            history,
            (user_id, request.chat_id) if request.chat_id else None,
            record if request.chat_id else None,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
