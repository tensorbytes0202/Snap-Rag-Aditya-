from langsmith import traceable
from rank_bm25 import BM25Okapi

from app.services.bm25_index import (
    load_index,
    stage_index,
    commit_index,
    discard_staged_index,
)


def _build_bm25(chunks):

    tokenized = [

        chunk["text"].split()

        for chunk in chunks

    ]

    return BM25Okapi(tokenized)


def stage_bm25_index(document_name, chunks) -> str:
    """
    Build a BM25 index and stage it (write to a temp file) WITHOUT
    replacing document_name's current index yet -- part of the
    two-phase swap used by the ingestion pipeline so a failure in a
    later step (e.g. storing vectors) never disturbs the existing
    index. Returns the staged temp path; the caller commits it with
    commit_bm25_index or discards it with discard_bm25_index.
    """

    bm25 = _build_bm25(chunks)

    return stage_index(document_name, bm25, chunks)


def commit_bm25_index(tmp_path: str, document_name: str) -> None:
    commit_index(tmp_path, document_name)


def discard_bm25_index(tmp_path: str) -> None:
    discard_staged_index(tmp_path)


@traceable(name="bm25_search", run_type="retriever")
def bm25_search(

    query,

    document_name,

    top_k=10

):

    data = load_index(document_name)

    if data is None:
        return []

    bm25 = data["bm25"]
    chunks = data["chunks"]

    scores = bm25.get_scores(
        query.split()
    )

    ranked = sorted(

        zip(chunks, scores),

        key=lambda x: x[1],

        reverse=True

    )

    return ranked[:top_k]