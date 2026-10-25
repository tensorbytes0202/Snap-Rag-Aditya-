import logging

from langsmith import traceable

from app.cache.cache_service import get_cache, set_cache
from app.core.config import settings
from app.core import metrics
from app.services.abrag import run_batches
from app.services.answering import make_attempt, present_answer
from app.services.followup_service import generate_followups
from app.services.memory_retriever import finalize_memory, recall_memory
from app.services.query_runner import run_retrieval
from app.services.response_builder import (
    build_failure_response,
    build_success_response,
)
from app.services.retrieval_pipeline import build_citations
from app.utils.parallel import run_parallel


_log = logging.getLogger(__name__)

# Part of every cache key: bump it when retrieval, prompts or verification
# change, so answers cached by the old pipeline are not served any more.
CACHE_VERSION = "3"


@traceable(name="rag_query", run_type="chain")
def process_query(
    question: str,
    active_document: str | None = None,
    history=None,
    session: tuple[str, str] | None = None,
):
    """`session` = (user_id, chat_id) of a chat: its verified snapshot
    memory is recalled and fused with the document context."""

    _log.info("RAG query started (document=%s)", active_document)

    # The cache key must include the active document, otherwise
    # "what is normalization?" on DBMS.pdf and on Android.pdf would
    # return the same cached response. With chat history the answer
    # depends on the conversation too, so the cache is skipped.
    cache_key = (
        f"v{CACHE_VERSION}::{active_document or 'ALL_DOCUMENTS'}"
        f"::{question.strip().lower()}"
    )

    use_cache = not history and not session

    cached_response = get_cache(cache_key) if use_cache else None

    if cached_response:
        _log.info("Returning response from Redis cache")
        metrics.inc("snaprag_queries_total", outcome="cache")
        return cached_response

    # Document retrieval and memory recall only share the question, so they
    # run side by side; the memory is then packed against the document context.
    found, memory = run_parallel(
        lambda: run_retrieval(question, active_document),
        (lambda: recall_memory(*session, question, active_document)) if session else (lambda: None),
    )

    if not found.ranked_chunks:
        metrics.inc("snaprag_queries_total", outcome="no_results")
        return build_failure_response(question, found.rewritten_question)

    memory = finalize_memory(memory, found.top_context)

    first = make_attempt(question, found, memory, history, "initial", with_followups=settings.FOLLOWUPS_ENABLED)

    best, attempts = run_batches(
        question,
        first,
        lambda **overrides: run_retrieval(question, active_document, **overrides),
        lambda deeper, strategy: make_attempt(question, deeper, memory, history, strategy),
    )

    followups = best.followups if best.followups is not None else (generate_followups(question, best.answer) if settings.FOLLOWUPS_ENABLED else [])

    _log.debug("Verification: %s", best.verification)

    found = best.found

    response = build_success_response(
        question=question,
        rewritten_question=found.rewritten_question,
        answer=present_answer(best),
        followups=followups,
        verification=best.verification,
        confidence=best.confidence,
        results=found.results,
        bm25_chunks=found.bm25_chunks,
        hybrid_chunks=found.hybrid_chunks,
        sources=found.sources,
        citations=build_citations(found.sources),
        memory=memory,
        attempts=attempts,
        signals=best.signals,
    )

    if use_cache:
        set_cache(cache_key, response)

    metrics.inc("snaprag_queries_total", outcome="answered" if best.supported else "unverified")

    return response
