import re

from langsmith import traceable
from rank_bm25 import BM25Okapi

from app.services.bm25_index import (
    load_index,
    stage_index,
    commit_index,
    discard_staged_index,
)


# Bump when `tokenize` changes: an index built with an older tokenizer is
# rebuilt from its stored chunks (no re-embedding, no API call) on first use.
TOKENIZER_VERSION = 2

_STOPWORDS = frozenset("""
a an and are as at be but by can could did do does for from had has have how if in
into is it its of on or such that the their then there these this those to was were
what when where which who whom why will with would not no nor so than too very
about above after again against all any because been before being below between both
down during each few further he her here him his i me more most my myself new now
only other our out over own same she should some them they through under until up
us we you your
""".split())


def tokenize(text):
    """Lowercase words and numbers without punctuation or stopwords, so
    "Negligence." in a page matches "negligence" in a query and filler words
    ("the", "using", "only") add no score."""

    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    ]


def _build_bm25(chunks):

    bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])

    bm25.tokenizer_version = TOKENIZER_VERSION

    return bm25


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

    if getattr(bm25, "tokenizer_version", 1) != TOKENIZER_VERSION:

        bm25 = _build_bm25(chunks)

        commit_index(stage_index(document_name, bm25, chunks), document_name)

    tokens = tokenize(query)

    if not tokens:
        return []

    scores = bm25.get_scores(tokens)

    # Zero-score chunks share no word with the query; keeping them would
    # pad the result with arbitrary chunks.
    ranked = sorted(

        ((chunk, score) for chunk, score in zip(chunks, scores) if score > 0),

        key=lambda x: x[1],

        reverse=True

    )

    return ranked[:top_k]