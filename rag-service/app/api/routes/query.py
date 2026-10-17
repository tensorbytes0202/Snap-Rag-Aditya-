import json
import logging
from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.exceptions import LLMProviderError
from app.core.security import get_current_user_id
from app.services.history import prompt_history
from app.services.query_service import process_query
from app.services.query_graph import stream_query
from app.services.document_registry import get_active_document, get_document_owner
from app.services.rate_limiter import QuotaExceeded, allow_request, consume, is_guest, refund
from app.services.session_store import append_turn, chat_exists
from app.services.snapshot_engine import after_turn


logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):

    question: str

    documents: List[str] = []

    # Optional chat (from POST /chats). With it, recent turns are shown to
    # the LLM and this turn is recorded; without it nothing is stored.
    chat_id: str | None = None


def _recent_history(user_id: str, chat_id: str | None) -> list[dict]:
    """The caller's turns for the prompt ([] without a chat)."""

    if chat_id is None:
        return []

    if not chat_exists(user_id, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")

    return prompt_history(user_id, chat_id)


def _reserve_question(user_id: str, active_document: str | None) -> None:
    """Demo guests need their own document selected and a question left;
    one is reserved here and refunded if answering fails."""

    if not allow_request(user_id):
        raise HTTPException(status_code=429, detail="Too many questions in a short time. Please wait a moment.")

    if is_guest(user_id) and (not active_document or get_document_owner(active_document) != user_id):
        raise HTTPException(status_code=400, detail="Upload a PDF first, then ask about it.")

    try:
        consume(user_id, "queries")
    except QuotaExceeded as exc:
        raise HTTPException(status_code=429, detail=str(exc))


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

    _reserve_question(user_id, active_document)

    try:
        response = process_query(
            request.question,
            active_document,
            history,
            (user_id, request.chat_id) if request.chat_id else None,
        )
    except LLMProviderError:
        refund(user_id, "queries")
        logger.exception("Language model request failed")
        raise HTTPException(
            status_code=502,
            detail="Language model request failed",
        )
    except BaseException:
        refund(user_id, "queries")
        raise

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


def _sse_events(question: str, active_document: str | None, history: list, session=None, record=None, on_failure=None):
    """`record(answer, citations, verified, confidence)` is called once a
    complete answer has streamed, to store the turn; `verified` and
    `confidence` are None if the check never ran. If no complete answer
    streamed, `on_failure()` is called instead (to refund a guest's question).

    A `revised` event replaces the answer streamed so far, so the record
    holds the revised one."""

    tokens, citations, failed = [], [], False
    verified = confidence = None
    revised = None

    try:
        for event in stream_query(question, active_document, history, session):
            if event["type"] == "token":
                tokens.append(event["token"])
            elif event["type"] == "citations":
                citations = event["citations"]
            elif event["type"] == "verification":
                verified, confidence = event.get("supported"), event.get("confidence")
            elif event["type"] == "revised":
                revised = event["answer"]
                citations = event.get("citations") or citations
                verified, confidence = event.get("supported"), event.get("confidence")
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

    if tokens and not failed:
        if record:
            record(revised if revised is not None else "".join(tokens), citations, verified, confidence)
    elif on_failure:
        on_failure()

    yield _sse({"type": "done"})


@router.post("/query/stream")
def query_stream(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):

    history = _recent_history(user_id, request.chat_id)

    active_document = get_active_document(user_id)

    _reserve_question(user_id, active_document)

    def record(answer: str, citations: list, verified: bool | None, confidence: float | None):
        append_turn(
            user_id, request.chat_id, request.question, answer,
            document=active_document, citations=citations,
            verified=verified, confidence=confidence,
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
            lambda: refund(user_id, "queries"),
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
