"""Query understanding and dense retrieval.

run_retrieval_pipeline turns a question into short search queries (see
query_planner), metadata filters and a retrieval config; retrieve_chunks runs
queries against Qdrant; build_sources and build_citations shape the results
for the API response.
"""

from langsmith import traceable

from app.repositories.vector_repository import search_similar_chunks
from app.services.embedding_service import generate_query_embedding
from app.services.hyde_service import generate_hyde
from app.services.metadata_parser import extract_metadata
from app.services.query_planner import plan_query
from app.services.query_router import get_retrieval_config


# A question this short with one topic is worth a HyDE passage; a long one
# already has one query per topic, and HyDE would add a call for each.
HYDE_MAX_WORDS = 40


def _as_float(value) -> float:

    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


# ==========================================================
# Retrieval Pipeline
# ==========================================================

@traceable(name="run_retrieval_pipeline", run_type="chain")
def run_retrieval_pipeline(question: str, force_hyde: bool = False):
    """One search query per topic, a page filter if the question names one
    ("on page 12"), and the retrieval config for its query type. The caller's
    active document (see query_runner) decides the document filter.
    `force_hyde` adds a HyDE passage whatever the question's size or type
    (corrective retrieval)."""

    plan = plan_query(question)

    subqueries = plan["subqueries"]

    metadata = extract_metadata(question) or {}
    metadata["document_name"] = None
    metadata.setdefault("page", None)

    config = get_retrieval_config(plan["query_type"])

    hyde = None

    if force_hyde:
        hyde = generate_hyde(" ".join(question.split()[:HYDE_MAX_WORDS]))
    elif config.get("use_hyde") and len(subqueries) == 1 and len(question.split()) <= HYDE_MAX_WORDS:
        hyde = generate_hyde(question)

    return {
        "query_type": plan["query_type"],
        "subqueries": subqueries,
        "hyde": hyde,
        "metadata": metadata,
        "config": config,
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


def build_citations(sources, top_k=10):
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
