"""Query understanding and dense retrieval.

run_retrieval_pipeline turns a question into search queries (rewrites,
sub-questions, HyDE passages), metadata filters and a retrieval config;
retrieve_chunks runs those queries against Qdrant; build_sources and
build_citations shape the results for the API response.
"""

import os

from langsmith import traceable

from app.repositories.vector_repository import search_similar_chunks
from app.services.embedding_service import generate_query_embedding
from app.services.hyde_service import generate_hyde
from app.services.metadata_parser import extract_metadata
from app.services.query_classifier import classify_query
from app.services.query_decomposition import decompose_query
from app.services.query_rewriter import rewrite_query
from app.services.query_router import get_retrieval_config
from app.services.self_query_service import generate_self_query


def _unique(queries) -> list[str]:
    """Non-empty strings, first occurrence kept."""

    return list(dict.fromkeys(q for q in queries if isinstance(q, str) and q.strip()))


def _as_float(value) -> float:

    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ==========================================================
# Retrieval Pipeline
# ==========================================================

@traceable(name="run_retrieval_pipeline", run_type="chain")
def run_retrieval_pipeline(question: str):

    # ------------------------------------------------------
    # Search queries: sub-questions, rewrites, the original
    # ------------------------------------------------------

    rewritten_question = rewrite_query(question)

    subqueries = decompose_query(question)

    if not isinstance(subqueries, list):
        subqueries = [subqueries]

    queries = list(subqueries)

    if isinstance(rewritten_question, list):
        queries.extend(rewritten_question)
    else:
        queries.append(rewritten_question)

    queries.append(question)

    # ------------------------------------------------------
    # Metadata filters: regex first, then the self-query LLM
    # for a document or page the regex missed. The caller's
    # active document (see query_runner) overrides document_name.
    # ------------------------------------------------------

    metadata = extract_metadata(question) or {}
    metadata.setdefault("document_name", None)
    metadata.setdefault("page", None)

    self_query = generate_self_query(question) or {}

    for key in ("document_name", "page"):
        if self_query.get(key):
            metadata[key] = self_query[key]

    if metadata["document_name"]:
        metadata["document_name"] = os.path.basename(str(metadata["document_name"])).strip()

    # ------------------------------------------------------
    # Query type -> retrieval config, HyDE passages
    # ------------------------------------------------------

    query_type = classify_query(question)

    config = get_retrieval_config(query_type)

    if config.get("use_hyde", False):
        queries.append(generate_hyde(question))
        queries.extend(generate_hyde(q) for q in subqueries if isinstance(q, str))

    return {
        "rewritten_question": rewritten_question,
        "metadata": metadata,
        "query_type": query_type,
        "config": config,
        "queries": _unique(queries),
    }


# ==========================================================
# Dense Retrieval
# ==========================================================

def retrieve_chunks(queries, metadata, config):
    """Dense search for every query, merged: each chunk keeps its best
    score, sorted best first."""

    metadata = metadata or {}

    best = {}

    for query in queries:

        if not query:
            continue

        hits = search_similar_chunks(
            generate_query_embedding(query),
            limit=config.get("dense_top_k", 10),
            document_name=metadata.get("document_name"),
            page=metadata.get("page"),
        )

        for hit in hits:

            payload = hit.payload or {}

            chunk = {
                "text": payload.get("text", ""),
                "document": payload.get("document_name", ""),
                "document_name": payload.get("document_name", ""),
                "page": payload.get("page"),
                "chunk_id": payload.get("chunk_id", ""),
                "parent_id": payload.get("parent_id", ""),
                "parent_text": payload.get("parent_text", ""),
                "child_id": payload.get("child_id", ""),
                "vector_score": float(hit.score),
                "bm25_score": None,
                "rerank_score": None,
                "retrieval_type": "dense",
            }

            key = chunk["chunk_id"] or (chunk["document"], chunk["page"], chunk["text"][:100])

            if key not in best or chunk["vector_score"] > best[key]["vector_score"]:
                best[key] = chunk

    return sorted(best.values(), key=lambda c: c["vector_score"], reverse=True)


# ==========================================================
# Sources & Citations
# ==========================================================

def build_sources(results):
    """One source per (document, page, parent): its best-scoring chunk,
    since `results` is sorted best first."""

    sources = []

    seen = set()

    for result in results:

        document = result.get("document_name") or result.get("document") or "Unknown"

        key = (document, result.get("page", "N/A"), result.get("parent_id", "N/A"))

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "document": document,
            "page": result.get("page", "N/A"),
            "chunk_id": result.get("chunk_id", "N/A"),
            "parent_id": result.get("parent_id", "N/A"),
            "text": result.get("text", ""),
            "parent_text": result.get("parent_text", ""),
            "vector_score": _as_float(result.get("vector_score")),
            "bm25_score": _as_float(result.get("bm25_score")),
            "rerank_score": _as_float(result.get("rerank_score")),
            "retrieval_type": result.get("retrieval_type", "dense"),
        })

    return sources


def build_citations(sources, top_k=5):
    """Up to `top_k` distinct (document, page) pairs, numbered from 1."""

    citations = []

    seen = set()

    for source in sources:

        document = source.get("document", "Unknown")
        page = source.get("page", "N/A")

        if (document, page) in seen:
            continue

        seen.add((document, page))

        citations.append({
            "source_id": len(citations) + 1,
            "document": document,
            "page": page,
        })

        if len(citations) >= top_k:
            break

    return citations
