from langsmith import traceable
import os

from app.services.query_rewriter import rewrite_query
from app.services.metadata_parser import extract_metadata
from app.services.query_classifier import classify_query
from app.services.query_router import get_retrieval_config
from app.services.hyde_service import generate_hyde
from app.services.embedding_service import (
    generate_query_embedding,
)

from app.repositories.vector_repository import (
    search_similar_chunks,
)

from app.services.context_compression import (
    compress_context,
)

from app.services.self_query_service import (
    generate_self_query,
)

from app.services.query_decomposition import (
    decompose_query,
)


# ==========================================================
# Retrieval Pipeline
# ==========================================================

@traceable(name="run_retrieval_pipeline", run_type="chain")
def run_retrieval_pipeline(question: str):

    # ======================================================
    # Rewrite Query
    # ======================================================

    rewritten_question = rewrite_query(question)

    # ======================================================
    # Query Decomposition
    # ======================================================

    subqueries = decompose_query(question)

    if not isinstance(subqueries, list):
        subqueries = [subqueries]

    # ======================================================
    # Build Queries
    # ======================================================

    queries = []

    queries.extend(subqueries)

    if isinstance(rewritten_question, str):

        if rewritten_question.strip():
            queries.append(
                rewritten_question
            )

    elif isinstance(
        rewritten_question,
        list,
    ):

        queries.extend(
            rewritten_question
        )

    # Fallback
    if not queries:

        queries = [
            question
        ]

    # Original question
    queries.append(
        question
    )

    # Remove duplicates
    queries = list(
        dict.fromkeys(
            q for q in queries
            if isinstance(q, str)
            and q.strip()
        )
    )

    # ======================================================
    # Metadata
    # ======================================================

    metadata = extract_metadata(
        question
    )

    # Make sure metadata is a dict
    if metadata is None:
        metadata = {}

    # Ensure keys exist
    metadata.setdefault(
        "document_name",
        None,
    )

    metadata.setdefault(
        "page",
        None,
    )

    metadata.setdefault(
        "chapter",
        None,
    )

    metadata.setdefault(
        "section",
        None,
    )

    # ======================================================
    # Self Query Retrieval
    # ======================================================

    self_query = generate_self_query(
        question
    )

    if self_query is None:
        self_query = {}

    self_query.setdefault(
        "document_name",
        None,
    )

    self_query.setdefault(
        "page",
        None,
    )

    self_query.setdefault(
        "chapter",
        None,
    )

    self_query.setdefault(
        "section",
        None,
    )

    # ======================================================
    # Document
    # ======================================================

    if self_query["document_name"]:

        metadata["document_name"] = (
            self_query["document_name"]
        )

    # Note: no fallback to a global "active document" here -- active
    # document is now per-user (see app.services.document_registry),
    # and process_query() already injects the caller's active
    # document into metadata["document_name"] after this pipeline
    # runs, so no fallback is needed here.

    # ======================================================
    # Page
    # ======================================================

    if self_query["page"]:

        metadata["page"] = (
            self_query["page"]
        )

    # ======================================================
    # Chapter
    # ======================================================

    if self_query["chapter"]:

        metadata["chapter"] = (
            self_query["chapter"]
        )

    # ======================================================
    # Section
    # ======================================================

    if self_query["section"]:

        metadata["section"] = (
            self_query["section"]
        )

    # ======================================================
    # Normalize Document Name
    # ======================================================

    if metadata.get("document_name"):

        metadata["document_name"] = (
            os.path.basename(
                str(
                    metadata["document_name"]
                )
            ).strip()
        )

    # ======================================================
    # Query Type
    # ======================================================

    query_type = classify_query(
        question
    )

    config = get_retrieval_config(
        query_type
    )

    # ======================================================
    # HyDE
    # ======================================================

    hyde_document = None

    if config.get(
        "use_hyde",
        False,
    ):

        hyde_document = generate_hyde(
            question
        )

        for subquery in subqueries:

            if not isinstance(
                subquery,
                str,
            ):
                continue

            hyde = generate_hyde(
                subquery
            )

            if hyde:
                queries.append(
                    hyde
                )

    if hyde_document:

        queries.append(
            hyde_document
        )

    # ======================================================
    # Remove Duplicate Queries
    # ======================================================

    queries = list(
        dict.fromkeys(
            q for q in queries
            if isinstance(q, str)
            and q.strip()
        )
    )

    return {

        "question": question,

        "rewritten_question":
            rewritten_question,

        "subqueries":
            subqueries,

        "metadata":
            metadata,

        "query_type":
            query_type,

        "config":
            config,

        "queries":
            queries,

        "hyde_document":
            hyde_document,

        "self_query":
            self_query,
    }


# ==========================================================
# Dense Retrieval
# ==========================================================

@traceable(name="retrieve_chunks", run_type="retriever")
def retrieve_chunks(
    queries,
    metadata,
    config,
):

    all_results = []

    document_name = (
        metadata.get(
            "document_name"
        )
        if metadata
        else None
    )

    page = (
        metadata.get("page")
        if metadata
        else None
    )

    # ======================================================
    # Search Each Query
    # ======================================================

    for query in queries:

        if not query:
            continue

        # --------------------------------------------------
        # Generate Query Embedding
        # --------------------------------------------------

        embedding = (
            generate_query_embedding(
                query
            )
        )

        # --------------------------------------------------
        # Qdrant Search
        # --------------------------------------------------

        results = search_similar_chunks(

            embedding,

            limit=config.get(
                "dense_top_k",
                10,
            ),

            document_name=
                document_name,

            page=page,
        )

        # --------------------------------------------------
        # Convert Qdrant ScoredPoint
        # to Dictionary
        # --------------------------------------------------

        for result in results:

            payload = (
                getattr(
                    result,
                    "payload",
                    None,
                )
                or {}
            )

            score = getattr(
                result,
                "score",
                0.0,
            )

            all_results.append({

                "text":
                    payload.get(
                        "text",
                        "",
                    ),

                "document":
                    payload.get(
                        "document_name",
                        "",
                    ),

                "document_name":
                    payload.get(
                        "document_name",
                        "",
                    ),

                "page":
                    payload.get(
                        "page",
                        None,
                    ),

                "chunk_id":
                    payload.get(
                        "chunk_id",
                        "",
                    ),

                "parent_id":
                    payload.get(
                        "parent_id",
                        "",
                    ),

                "parent_text":
                    payload.get(
                        "parent_text",
                        "",
                    ),

                "child_id":
                    payload.get(
                        "child_id",
                        "",
                    ),

                "vector_score":
                    float(score),

                "bm25_score":
                    None,

                "rerank_score":
                    None,

                "retrieval_type":
                    "dense",
            })

    # ======================================================
    # Remove Duplicate Chunks
    # ======================================================

    unique = {}

    for result in all_results:

        chunk_id = result.get(
            "chunk_id"
        )

        # If chunk_id is missing,
        # use combination as fallback
        if not chunk_id:

            chunk_id = (
                result.get(
                    "document",
                    "",
                ),
                result.get(
                    "page",
                    "",
                ),
                result.get(
                    "text",
                    "",
                )[:100],
            )

        # --------------------------------------------------
        # Keep Highest Vector Score
        # --------------------------------------------------

        if chunk_id not in unique:

            unique[
                chunk_id
            ] = result

        else:

            current_score = (
                result.get(
                    "vector_score",
                    0.0,
                )
            )

            previous_score = (
                unique[
                    chunk_id
                ].get(
                    "vector_score",
                    0.0,
                )
            )

            if current_score > previous_score:

                unique[
                    chunk_id
                ] = result

    results = list(
        unique.values()
    )

    # ======================================================
    # Sort By Vector Score
    # ======================================================

    results.sort(

        key=lambda x:
            x.get(
                "vector_score",
                0.0,
            ),

        reverse=True,
    )

    return results


# ==========================================================
# Parent Chunks
# ==========================================================

def _get_parent_chunks(
    results,
):

    parents = []

    seen = set()

    for result in results:

        # --------------------------------------------------
        # Dictionary result
        # --------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            parent_id = (
                result.get(
                    "parent_id"
                )
            )

            parent_text = (
                result.get(
                    "parent_text"
                )
            )

        # --------------------------------------------------
        # Qdrant object fallback
        # --------------------------------------------------

        else:

            payload = (
                getattr(
                    result,
                    "payload",
                    None,
                )
                or {}
            )

            parent_id = (
                payload.get(
                    "parent_id"
                )
            )

            parent_text = (
                payload.get(
                    "parent_text"
                )
            )

        # --------------------------------------------------
        # Ignore empty parents
        # --------------------------------------------------

        if not parent_text:
            continue

        # If parent ID doesn't exist,
        # use parent text as unique key
        unique_id = (
            parent_id
            if parent_id
            else parent_text
        )

        if unique_id in seen:
            continue

        seen.add(
            unique_id
        )

        parents.append(
            parent_text
        )

    return parents


# ==========================================================
# Context Expansion
# ==========================================================

def _expand_context(
    results,
    window_size=1,
):

    expanded_context = []

    seen = set()

    for result in results:

        # --------------------------------------------------
        # Dictionary result
        # --------------------------------------------------

        if isinstance(
            result,
            dict,
        ):

            parent_text = (
                result.get(
                    "parent_text"
                )
            )

            text = (
                result.get(
                    "text"
                )
            )

        # --------------------------------------------------
        # Qdrant object fallback
        # --------------------------------------------------

        else:

            payload = (
                getattr(
                    result,
                    "payload",
                    None,
                )
                or {}
            )

            parent_text = (
                payload.get(
                    "parent_text"
                )
            )

            text = (
                payload.get(
                    "text"
                )
            )

        # --------------------------------------------------
        # Prefer parent context
        # --------------------------------------------------

        context_text = (
            parent_text
            if parent_text
            else text
        )

        if not context_text:
            continue

        if context_text in seen:
            continue

        seen.add(
            context_text
        )

        expanded_context.append(
            context_text
        )

    return expanded_context


# ==========================================================
# Build Retrieval Context
# ==========================================================

@traceable(name="build_context", run_type="chain")
def build_context(
    question,
    results,
    config,
):

    # ======================================================
    # Parent Retrieval
    # ======================================================

    parent_chunks = (
        _get_parent_chunks(
            results
        )
    )

    # ======================================================
    # Context Expansion
    # ======================================================

    expanded_chunks = (
        _expand_context(
            results,

            window_size=config.get(
                "window_size",
                1,
            ),
        )
    )

    # ======================================================
    # Context Compression
    # ======================================================

    compression_top_k = config.get(
        "compression_top_k",
        8,
    )

    compressed_chunks = (
        compress_context(

            question,

            expanded_chunks,

            top_k=
                compression_top_k,
        )
    )

    # ======================================================
    # Safety
    # ======================================================

    if compressed_chunks is None:

        compressed_chunks = []

    # ======================================================
    # Final Context
    # ======================================================

    context = "\n\n".join(

        str(chunk)

        for chunk
        in compressed_chunks

        if chunk
    )

    return {

        "parent_chunks":
            parent_chunks,

        "expanded_chunks":
            expanded_chunks,

        "compressed_chunks":
            compressed_chunks,

        "context":
            context,
    }


# ==========================================================
# Build Context Chunks
# ==========================================================

def build_context_chunks(
    ranked_chunks,
    top_k=5,
):

    context_chunks = []

    for chunk in ranked_chunks[:top_k]:

        # --------------------------------------------------
        # Dictionary format
        # --------------------------------------------------

        if isinstance(
            chunk,
            dict,
        ):

            context_chunks.append({

                "text":
                    chunk.get(
                        "text",
                        "",
                    ),

                "page":
                    chunk.get(
                        "page",
                        None,
                    ),

                "document":
                    chunk.get(
                        "document_name"
                    )
                    or chunk.get(
                        "document"
                    ),

                "chunk_id":
                    chunk.get(
                        "chunk_id"
                    ),

                "parent_id":
                    chunk.get(
                        "parent_id"
                    ),

                "vector_score":
                    chunk.get(
                        "vector_score"
                    ),

                "bm25_score":
                    chunk.get(
                        "bm25_score"
                    ),

                "retrieval_type":
                    chunk.get(
                        "retrieval_type"
                    ),

                "rerank_score":
                    chunk.get(
                        "rerank_score"
                    ),

            })

            continue

        # --------------------------------------------------
        # Qdrant object fallback
        # --------------------------------------------------

        payload = (
            getattr(
                chunk,
                "payload",
                None,
            )
            or {}
        )

        context_chunks.append({

            "text":
                payload.get(
                    "text",
                    "",
                ),

            "page":
                payload.get(
                    "page",
                    None,
                ),

            "document":
                payload.get(
                    "document_name",
                    "",
                ),

            "chunk_id":
                payload.get(
                    "chunk_id"
                ),

            "parent_id":
                payload.get(
                    "parent_id"
                ),

            "vector_score":
                getattr(
                    chunk,
                    "score",
                    None,
                ),

            "bm25_score":
                None,

            "retrieval_type":
                "dense",

            "rerank_score":
                None,
        })

    return context_chunks


# ==========================================================
# Build Sources
# ==========================================================

def build_sources(
    results,
):

    sources = []

    seen = set()

    for result in results:

        # ==================================================
        # Dictionary Result
        # ==================================================

        if isinstance(
            result,
            dict,
        ):

            document = (
                result.get(
                    "document_name"
                )
                or result.get(
                    "document"
                )
                or "Unknown"
            )

            page = result.get(
                "page",
                "N/A",
            )

            parent_id = result.get(
                "parent_id",
                "N/A",
            )

            chunk_id = result.get(
                "chunk_id",
                "N/A",
            )

            text = result.get(
                "text",
                "",
            )

            parent_text = result.get(
                "parent_text",
                "",
            )

            vector_score = result.get(
                "vector_score",
                0.0,
            )

            retrieval_type = result.get(
                "retrieval_type",
                "dense",
            )

            bm25_score = result.get(
                "bm25_score",
                0.0,
            )

            rerank_score = result.get(
                "rerank_score",
                0.0,
            )

        # ==================================================
        # Qdrant Object Fallback
        # ==================================================

        else:

            payload = (
                getattr(
                    result,
                    "payload",
                    None,
                )
                or {}
            )

            document = payload.get(
                "document_name",
                "Unknown",
            )

            page = payload.get(
                "page",
                "N/A",
            )

            parent_id = payload.get(
                "parent_id",
                "N/A",
            )

            chunk_id = payload.get(
                "chunk_id",
                "N/A",
            )

            text = payload.get(
                "text",
                "",
            )

            parent_text = payload.get(
                "parent_text",
                "",
            )

            vector_score = getattr(
                result,
                "score",
                0.0,
            )

            retrieval_type = "dense"

            bm25_score = 0.0

            rerank_score = 0.0

        # ==================================================
        # Normalize Scores
        # ==================================================

        try:

            vector_score = float(
                vector_score
                if vector_score is not None
                else 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            vector_score = 0.0

        try:

            bm25_score = float(
                bm25_score
                if bm25_score is not None
                else 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            bm25_score = 0.0

        try:

            rerank_score = float(
                rerank_score
                if rerank_score is not None
                else 0.0
            )

        except (
            TypeError,
            ValueError,
        ):

            rerank_score = 0.0

        # ==================================================
        # Deduplicate Source
        # ==================================================

        key = (

            document,

            page,

            parent_id,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        # ==================================================
        # Build Source
        # ==================================================

        sources.append({

            "document":
                document,

            "page":
                page,

            "chunk_id":
                chunk_id,

            "parent_id":
                parent_id,

            "text":
                text,

            "parent_text":
                parent_text,

            "vector_score":
                vector_score,

            "bm25_score":
                bm25_score,

            "rerank_score":
                rerank_score,

            "retrieval_type":
                retrieval_type,
        })

    return sources


# ==========================================================
# Build Citations
# ==========================================================

def build_citations(
    sources,
    top_k=5,
):

    citations = []

    seen = set()

    for source in sources:

        key = (

            source.get(
                "document",
                "Unknown",
            ),

            source.get(
                "page",
                "N/A",
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        citations.append({

            "source_id":
                len(citations) + 1,

            "document":
                source.get(
                    "document",
                    "Unknown",
                ),

            "page":
                source.get(
                    "page",
                    "N/A",
                ),
        })

        if len(citations) >= top_k:
            break

    return citations


# ==========================================================
# Convert BM25 Result
# ==========================================================

def convert_bm25_chunks(
    bm25_results,
):

    chunks = []

    if not bm25_results:
        return chunks

    for item in bm25_results:

        # ==================================================
        # Expected format:
        #
        # (chunk, score)
        # ==================================================

        if (
            isinstance(
                item,
                tuple,
            )
            and len(item) == 2
        ):

            chunk, score = item

        else:

            # Skip malformed result
            continue

        # --------------------------------------------------
        # Make a copy
        # --------------------------------------------------

        if isinstance(
            chunk,
            dict,
        ):

            chunk = dict(
                chunk
            )

        else:

            # Qdrant object fallback
            payload = (
                getattr(
                    chunk,
                    "payload",
                    None,
                )
                or {}
            )

            chunk = {

                "text":
                    payload.get(
                        "text",
                        "",
                    ),

                "document":
                    payload.get(
                        "document_name",
                        "",
                    ),

                "document_name":
                    payload.get(
                        "document_name",
                        "",
                    ),

                "page":
                    payload.get(
                        "page",
                        None,
                    ),

                "chunk_id":
                    payload.get(
                        "chunk_id",
                        "",
                    ),

                "parent_id":
                    payload.get(
                        "parent_id",
                        "",
                    ),

                "parent_text":
                    payload.get(
                        "parent_text",
                        "",
                    ),
            }

        # --------------------------------------------------
        # BM25 Metadata
        # --------------------------------------------------

        chunk[
            "retrieval_type"
        ] = "bm25"

        try:

            chunk[
                "bm25_score"
            ] = float(
                score
            )

        except (
            TypeError,
            ValueError,
        ):

            chunk[
                "bm25_score"
            ] = 0.0

        # Dense score not available
        chunk.setdefault(
            "vector_score",
            0.0,
        )

        chunk.setdefault(
            "rerank_score",
            None,
        )

        chunks.append(
            chunk
        )

    return chunks