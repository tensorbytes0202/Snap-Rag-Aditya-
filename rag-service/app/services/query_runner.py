"""Retrieval flow shared by the batch and the streaming query:

    plan -> for every search query: dense search + BM25 -> RRF fusion -> rerank
         -> merge the per-query rankings round-robin

`on_step(message, **detail)` is an optional progress callback. The streaming
path uses it to report progress; the batch path leaves it out.
"""

import logging
from dataclasses import dataclass, field, replace

from app.services.bm_25_service import bm25_search
from app.services.document_registry import get_document_overview
from app.services.hyde_service import generate_hyde
from app.services.query_planner import MAX_SUBQUERIES
from app.services.reranker_services import rerank_chunks
from app.services.retrieval_pipeline import (
    HYDE_MAX_WORDS,
    build_sources,
    retrieve_chunks,
    run_retrieval_pipeline,
)
from app.services.rrf_service import rrf_fusion


_log = logging.getLogger(__name__)

# A question with several topics gets one ranking per topic. Each topic
# contributes its PER_QUERY best chunks, so one dominant topic cannot crowd
# the others out; MAX_PASSAGES bounds the context handed to the LLM. A
# one-topic question reads the first batch its intent asks for (query_router).
PER_QUERY = 3
MAX_PASSAGES = 18

# Fused candidates per query that the cross-encoder actually scores.
RERANK_POOL = 20

# Most candidates a wide search asks each index for.
WIDE_LIMIT = 40

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
    limit: int = 5
    # The query plan, so a retry can reuse it instead of planning again.
    plan: dict = field(default_factory=dict)

    def deeper(self, step: int) -> "Retrieval | None":
        """The next batch: the same ranking read `step` passages further (no new
        search). None when the ranking or MAX_PASSAGES is used up."""

        limit = min(MAX_PASSAGES, len(self.ranked_chunks), self.limit + step)

        if limit <= self.limit:
            return None

        deeper = replace(self, limit=limit)
        deeper.sources = build_sources(deeper.top_chunks)

        return deeper

    @property
    def top_chunks(self) -> list:
        return self.ranked_chunks[:self.limit]

    @property
    def top_context(self) -> str:
        """Small-to-big: children are matched, but the LLM reads each top
        chunk's parent, so facts split across sibling children (e.g. two
        screenshots on one page) arrive together. Parents are deduplicated
        in rank order; a chunk without a parent falls back to its own text."""

        passages = list(dict.fromkeys(
            f"[Source: {_document(chunk)}, PDF page {chunk.get('page')}]\n"
            + (chunk.get("parent_text") or chunk["text"])
            for chunk in self.top_chunks
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


def _parent_key(chunk: dict) -> tuple:

    return (_document(chunk), chunk.get("page"), chunk.get("parent_id"))


def _merge_rankings(rankings: list[list]) -> list:
    """Round-robin over the per-query rankings (best of every query first),
    keeping one chunk per parent passage."""

    merged = []

    seen = set()

    for rank in range(max(map(len, rankings), default=0)):

        for ranking in rankings:

            if rank >= len(ranking):
                continue

            chunk = ranking[rank]

            if _parent_key(chunk) in seen:
                continue

            seen.add(_parent_key(chunk))

            merged.append(chunk)

    return merged


def run_retrieval(
    question: str,
    active_document: str | None = None,
    on_step=None,
    *,
    plan: dict | None = None,
    extra_queries=(),
    wide: bool = False,
) -> Retrieval:
    """Stops early when nothing is found: `results` is empty if dense search
    found nothing, `ranked_chunks` is empty if nothing survived reranking.

    The keyword arguments are for a fresh search when the confidence check wants
    more evidence than the current ranking holds (see abrag): `plan` reuses an
    earlier plan, `extra_queries` adds search queries, `wide` searches twice as
    deep, reads more passages and adds a HyDE passage."""

    step = on_step or (lambda message, **detail: None)

    if plan is None:
        pipeline = run_retrieval_pipeline(question, force_hyde=wide)
    elif wide and not plan.get("hyde"):
        pipeline = {**plan, "hyde": generate_hyde(" ".join(question.split()[:HYDE_MAX_WORDS]))}
    else:
        pipeline = plan

    metadata = dict(pipeline["metadata"])
    subqueries = list(dict.fromkeys([*pipeline["subqueries"], *extra_queries]))[:MAX_SUBQUERIES]
    spread = 2 if wide else 1
    config = {
        **pipeline["config"],
        "dense_top_k": min(WIDE_LIMIT, spread * max(pipeline["config"]["dense_top_k"], MIN_DENSE_CANDIDATES)),
        "bm25_top_k": min(WIDE_LIMIT, spread * max(pipeline["config"]["bm25_top_k"], MIN_BM25_CANDIDATES)),
    }

    if active_document:
        metadata["document_name"] = active_document

    found = Retrieval(subqueries, pipeline["query_type"], config, plan=pipeline)

    if active_document:
        found.overview = get_document_overview(active_document)

    step("Understood the question", rewritten=found.rewritten_question, kind=found.query_type)

    step("Running dense + keyword retrieval", queries=len(subqueries))

    rankings = []

    for subquery in subqueries:

        # HyDE (short questions only) widens the dense search, never BM25.
        dense = retrieve_chunks(
            [subquery, pipeline["hyde"]] if pipeline["hyde"] else [subquery], metadata, config
        )

        if active_document:
            dense = [r for r in dense if _document(r) == active_document]

        found.results = found.results + [r for r in dense if r not in found.results]

        dense_chunks = [
            {**{key: result.get(key) for key in _DENSE_KEYS}, "retrieval_type": "dense"}
            for result in dense
        ]

        bm25_chunks = [
            {**chunk, "retrieval_type": "bm25", "bm25_score": float(score)}
            for chunk, score in bm25_search(
                query=subquery,
                document_name=metadata.get("document_name"),
                top_k=config["bm25_top_k"],
            )
            if not active_document or _document(chunk) == active_document
        ]

        found.bm25_chunks += bm25_chunks

        hybrid = rrf_fusion(dense_chunks, bm25_chunks)

        found.hybrid_chunks += hybrid

        rankings.append(rerank_chunks(subquery, hybrid[:RERANK_POOL * spread]))

    if not found.results:
        _log.info("No dense results for active document: %s", active_document)
        step("Dense retrieval found nothing")
        return found

    step("Dense retrieval found chunks", count=len(found.results))

    step("Re-ranking with cross-encoder")

    found.ranked_chunks = _merge_rankings(rankings)

    per_query = PER_QUERY if len(subqueries) > 1 else config["passages"]
    found.limit = min(MAX_PASSAGES, (per_query + (2 if wide else 0)) * len(subqueries))

    # Citations name the pages the LLM actually reads, not every dense hit.
    found.sources = build_sources(found.top_chunks)

    _log.info(
        "Queries=%d Dense=%d BM25=%d Hybrid=%d Reranked=%d Context=%d",
        len(subqueries), len(found.results), len(found.bm25_chunks),
        len(found.hybrid_chunks), len(found.ranked_chunks), len(found.top_chunks),
    )

    return found
