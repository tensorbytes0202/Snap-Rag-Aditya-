import logging

from langsmith import traceable

from app.cache.cache_service import get_cache, set_cache
from app.services.confidence_service import calculate_confidence
from app.services.followup_service import generate_followups
from app.services.generation_service import generate_answer
from app.services.memory_retriever import memory_block, recall_memory
from app.services.query_runner import run_retrieval
from app.services.response_builder import (
    build_failure_response,
    build_success_response,
)
from app.services.retrieval_pipeline import build_citations
from app.services.verification_service import verify_answer


_log = logging.getLogger(__name__)

UNVERIFIED_ANSWER = (
    "⚠️ The generated answer "
    "could not be fully verified "
    "from the retrieved document."
)


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
        f"{active_document or 'ALL_DOCUMENTS'}"
        f"::{question.strip().lower()}"
    )

    use_cache = not history and not session

    cached_response = get_cache(cache_key) if use_cache else None

    if cached_response:
        _log.info("Returning response from Redis cache")
        return cached_response

    found = run_retrieval(question, active_document)

    if not found.ranked_chunks:
        return build_failure_response(question, found.rewritten_question)

    memory = recall_memory(*session, question, active_document, found.top_context) if session else None

    # Fused context: documents plus verified memory. The answer check gets
    # the same context, so an answer drawn from memory can be verified.
    top_context = found.top_context + memory_block(memory)

    answer = generate_answer(question, top_context, found.query_type, history)

    followups = generate_followups(question, answer)

    verification = verify_answer(question, answer, top_context)

    _log.debug("Verification: %s", verification)

    if not verification.get("supported", False):
        answer = UNVERIFIED_ANSWER

    response = build_success_response(
        question=question,
        rewritten_question=found.rewritten_question,
        answer=answer,
        followups=followups,
        verification=verification,
        confidence=calculate_confidence(
            [chunk["rerank_score"] for chunk in found.top_chunks]
        ),
        results=found.results,
        bm25_chunks=found.bm25_chunks,
        hybrid_chunks=found.hybrid_chunks,
        sources=found.sources,
        citations=build_citations(found.sources),
        memory=memory,
    )

    if use_cache:
        set_cache(cache_key, response)

    return response
