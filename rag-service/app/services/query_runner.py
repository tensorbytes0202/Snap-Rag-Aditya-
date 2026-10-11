"""Retrieval flow shared by the batch and the streaming query:

    understand -> dense search -> BM25 -> RRF fusion -> rerank

`on_step(message, **detail)` is an optional progress callback. The streaming
path uses it to report progress; the batch path leaves it out.
"""

import logging
from dataclasses import dataclass, field

from app.services.bm_25_service import bm25_search
from app.services.document_registry import get_document_overview
from app.services.reranker_services import rerank_chunks
from app.services.retrieval_pipeline import (
    build_sources,
    retrieve_chunks,
    run_retrieval_pipeline,
)
from app.services.rrf_service import rrf_fusion


_log = logging.getLogger(__name__)

TOP_K = 5

# Smallest candidate pool handed to the cross-encoder. Dense similarity is
# only a rough first cut (a "test succeeded" page can score below every
# config page for "what proves it works?"), so retrieve wide and let the
# reranker pick the TOP_K. Reranking ~30 short chunks costs well under 1s.
MIN_DENSE_CANDIDATES = 20
MIN_BM25_CANDIDATES = 10

_DENSE_KEYS = ("text", "parent_text", "page", "document", "chunk_id", "parent_id", "vector_score")


@dataclass
class Retrieval:

    rewritten_question: str
    query_type: str
    config: dict
    results: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    bm25_chunks: list = field(default_factory=list)
    hybrid_chunks: list = field(default_factory=list)
    ranked_chunks: list = field(default_factory=list)
    overview: list = field(default_factory=list)

    @property
    def top_chunks(self) -> list:
        return self.ranked_chunks[:TOP_K]

    @property
    def top_context(self) -> str:
        """Small-to-big: children are matched, but the LLM reads each top
        chunk's parent, so facts split across sibling children (e.g. two
        screenshots on one page) arrive together. Parents are deduplicated
        in rank order; a chunk without a parent falls back to its own text."""

        passages = list(dict.fromkeys(
            chunk.get("parent_text") or chunk["text"] for chunk in self.top_chunks
        ))

        # One gist line per page of the active document, so a question
        # whose answer is a page's *meaning* ("what proves it works?")
        # can be answered even when that page didn't rank in the top K.
        if self.overview:
            passages.append(
                "Document overview (one line per page):\n" + "\n".join(self.overview)
            )

        return "\n\n".join(passages)


def _document(item: dict):

    metadata = item.get("metadata", {})

    return (
        item.get("document")
        or item.get("document_name")
        or metadata.get("document")
        or metadata.get("document_name")
    )


def run_retrieval(
    question: str,
    active_document: str | None = None,
    on_step=None,
) -> Retrieval:
    """Stops early when nothing is found: `results` is empty if dense search
    found nothing, `ranked_chunks` is empty if nothing survived reranking."""

    step = on_step or (lambda message, **detail: None)

    pipeline = run_retrieval_pipeline(question)
    metadata = pipeline["metadata"]
    config = {
        **pipeline["config"],
        "dense_top_k": max(pipeline["config"]["dense_top_k"], MIN_DENSE_CANDIDATES),
        "bm25_top_k": max(pipeline["config"]["bm25_top_k"], MIN_BM25_CANDIDATES),
    }

    if active_document:
        metadata["document_name"] = active_document

    found = Retrieval(pipeline["rewritten_question"], pipeline["query_type"], config)

    if active_document:
        found.overview = get_document_overview(active_document)

    step("Understood the question", rewritten=found.rewritten_question, kind=found.query_type)

    step("Running dense vector retrieval", queries=len(pipeline["queries"]))

    results = retrieve_chunks(pipeline["queries"], metadata, config)

    if active_document:
        results = [r for r in results if _document(r) == active_document]

    if not results:
        _log.info("No dense results for active document: %s", active_document)
        step("Dense retrieval found nothing")
        return found

    step("Dense retrieval found chunks", count=len(results))

    found.results = results
    found.sources = build_sources(results)

    # Fuse every dense hit, not `sources`: build_sources keeps one chunk per
    # parent (i.e. per page, for short pages), which is right for citations
    # but would drop all but the best chunk of each page before reranking.
    dense_chunks = [
        {**{key: result.get(key) for key in _DENSE_KEYS}, "retrieval_type": "dense"}
        for result in results
    ]

    step("Running BM25 keyword search")

    found.bm25_chunks = [
        {**chunk, "retrieval_type": "bm25", "bm25_score": float(score)}
        for chunk, score in bm25_search(
            query=question,
            document_name=metadata.get("document_name"),
            top_k=config["bm25_top_k"],
        )
        if not active_document or _document(chunk) == active_document
    ]

    step("Fusing dense + BM25 results (RRF)")
    found.hybrid_chunks = rrf_fusion(dense_chunks, found.bm25_chunks)

    step("Re-ranking with cross-encoder")
    found.ranked_chunks = rerank_chunks(question, found.hybrid_chunks)

    _log.info(
        "Dense=%d BM25=%d Hybrid=%d Reranked=%d",
        len(dense_chunks), len(found.bm25_chunks),
        len(found.hybrid_chunks), len(found.ranked_chunks),
    )

    return found
