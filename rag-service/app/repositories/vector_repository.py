import logging
import uuid

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import settings
from app.api.dependencies.qdrant import qdrant_client


client = qdrant_client

COLLECTION_NAME = settings.COLLECTION_NAME

_log = logging.getLogger(__name__)

# Tags every point stored for a document with the ingestion run that
# wrote it, so store_embeddings can write a new batch in full before
# removing whatever batch(es) came before it -- see store_embeddings.
BATCH_ID_KEY = "batch_id"


# ==========================================================
# Create Collection
# ==========================================================

def create_collection():

    _log.info("Connecting to Qdrant...")

    collections = client.get_collections()

    existing = [
        collection.name
        for collection in collections.collections
    ]

    if COLLECTION_NAME not in existing:

        client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config=VectorParams(

                size=settings.EMBEDDING_DIM,

                distance=Distance.COSINE

            )

        )

        _log.info("Collection '%s' created.", COLLECTION_NAME)

    else:

        _log.info("Collection '%s' already exists.", COLLECTION_NAME)


# ==========================================================
# Store Embeddings
# ==========================================================
#
# Writes a fresh, uniquely-tagged batch of points for `document_name`
# and only then removes whatever was stored for it before -- so a
# reader querying this document never sees zero vectors for it, and a
# failure either leaves the previous batch completely untouched (if
# writing the new one failed) or is rolled back on a best-effort basis
# (if removing the previous one failed after the new one was written).
#
# `pre_existing` (passed down from document_service.ingest_document,
# which already knows whether this name was registered before this
# call) controls what happens if removing the *old* batch fails: for a
# re-upload (True) there's a previously-working document to protect,
# so a client.count() check decides whether to trust the swap already
# happened or roll back (see below); for a first upload (False) there
# is nothing old to lose, so any failure here always rolls back the
# new batch without needing to check anything.

def store_embeddings(

    chunks,

    embeddings,

    document_name,

    pre_existing=False,

):

    if not chunks or not embeddings:

        _log.info("No chunks or embeddings to store for %s; skipping.", document_name)

        return

    batch_id = uuid.uuid4().hex

    points = []

    for chunk, embedding in zip(chunks, embeddings):

        points.append(

            PointStruct(

                id=str(uuid.uuid4()),

                vector=embedding,

                payload={

                    "chunk_id": str(uuid.uuid4()),

                    "parent_id": chunk["parent_id"],

                    "child_id": chunk["child_id"],

                    "parent_text": chunk["parent_text"],

                    "text": chunk["text"],

                    "document_name": chunk["document"],

                    "page": chunk["page"],

                    BATCH_ID_KEY: batch_id,

                }

            )

        )

    batch_size = 100

    total_batches = (

        len(points) + batch_size - 1

    ) // batch_size

    _log.info(
        "Storing %d point(s) for %s across %d batch(es) (batch_id=%s)",
        len(points), document_name, total_batches, batch_id,
    )

    # ------------------------------------------------------
    # Write the new vectors first. If any batch fails, ALWAYS try to
    # remove the new batch -- even if no batch was "confirmed"
    # written by the client. With wait=True (the client default), a
    # read timeout or connection drop can happen after the server has
    # already applied the upsert, so "the call raised" does not mean
    # "nothing was written". _delete_batch is idempotent (a filter
    # matching zero points is a no-op), so this is always safe, and it
    # closes the gap where a failed first upload could otherwise leave
    # orphan vectors with no registry entry pointing at them.
    # ------------------------------------------------------

    try:

        for i in range(0, len(points), batch_size):

            batch = points[i:i + batch_size]

            client.upsert(

                collection_name=COLLECTION_NAME,

                points=batch

            )

            _log.info("Uploaded batch %d/%d", (i // batch_size) + 1, total_batches)

    except Exception:

        _log.exception(
            "Failed to upsert new vectors for %s (batch_id=%s); removing any batch "
            "that may have been written server-side despite the error",
            document_name, batch_id,
        )

        _rollback_new_batch(document_name, batch_id)

        raise

    # ------------------------------------------------------
    # Now remove everything else stored for this document (its
    # previous batch, or nothing on a first upload).
    #
    # On a first upload (pre_existing=False) there is no previously-
    # working document to protect, so ANY failure here always rolls
    # back the new batch -- no need to check anything further.
    #
    # On a re-upload (pre_existing=True), first check whether the
    # delete actually applied anyway (the same client-sees-an-error-
    # but-the-server-already-did-it gap as above): if no old points
    # remain, the swap already committed and rolling back would leave
    # the document registered with zero vectors, which is worse than a
    # client-side error we can ignore. Only roll back when old points
    # are confirmed to still be there -- and if even confirming that
    # fails (a double failure), fall back to rolling back anyway: we
    # can't tell which state we're in, and duplicate vectors are a
    # smaller risk to leave unresolved than a zeroed-out document.
    # ------------------------------------------------------

    old_points_filter = Filter(
        must=[
            FieldCondition(
                key="document_name",
                match=MatchValue(value=document_name),
            )
        ],
        must_not=[
            FieldCondition(
                key=BATCH_ID_KEY,
                match=MatchValue(value=batch_id),
            )
        ],
    )

    try:

        client.delete(

            collection_name=COLLECTION_NAME,

            points_selector=FilterSelector(filter=old_points_filter),

        )

    except Exception:

        if not pre_existing:

            _log.exception(
                "Failed to finish the vector swap for %s (batch_id=%s) on a "
                "first upload; rolling back the new batch",
                document_name, batch_id,
            )
            _rollback_new_batch(document_name, batch_id)
            raise

        try:
            remaining_old = client.count(
                collection_name=COLLECTION_NAME,
                count_filter=old_points_filter,
                exact=True,
            ).count

        except Exception:

            _log.exception(
                "Failed to remove previous vectors for %s (batch_id=%s), and "
                "confirming whether any remain also failed; rolling back the "
                "new batch as a best-effort safety default",
                document_name, batch_id,
            )
            _rollback_new_batch(document_name, batch_id)
            raise

        if remaining_old == 0:

            _log.warning(
                "Removing previous vectors for %s reported an error, but none "
                "remain -- the delete applied server-side despite the client-side "
                "error; treating the swap as complete.",
                document_name,
            )

        else:

            _log.exception(
                "Failed to remove previous vectors for %s (batch_id=%s); "
                "rolling back the new batch",
                document_name, batch_id,
            )
            _rollback_new_batch(document_name, batch_id)
            raise

    _log.info("Embedding upload complete for %s", document_name)


def _rollback_new_batch(document_name, batch_id):
    """Best-effort delete of the new batch; logs (rather than raising)
    if the rollback attempt itself also fails, so the caller's own
    exception is the one that propagates."""

    try:
        _delete_batch(document_name, batch_id)
    except Exception:
        _log.exception(
            "Rollback of the new vector batch for %s (batch_id=%s) also failed; "
            "duplicate vectors may remain until the next re-upload",
            document_name, batch_id,
        )


def _delete_batch(document_name, batch_id):
    """Remove just the points written for `batch_id` -- used to clean
    up/roll back a batch that must not survive. Idempotent: deleting
    with a filter that matches nothing is a no-op."""

    client.delete(

        collection_name=COLLECTION_NAME,

        points_selector=FilterSelector(

            filter=Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name),
                    ),
                    FieldCondition(
                        key=BATCH_ID_KEY,
                        match=MatchValue(value=batch_id),
                    ),
                ]
            )

        ),

    )


# ==========================================================
# Dense Retrieval
# ==========================================================

def search_similar_chunks(

    query_embedding,

    limit=10,

    document_name=None,

    page=None   

):

    conditions = []

    if document_name:

        conditions.append(

            FieldCondition(

                key="document_name",

                match=MatchValue(

                    value=document_name.strip()

                )

            )

        )

    if page is not None:

        conditions.append(

            FieldCondition(

                key="page",

                match=MatchValue(

                    value=page

                )

            )

        )

    query_filter = None

    if conditions:

        query_filter = Filter(

            must=conditions

        )

    _log.info(
        "Search request: document=%s page=%s limit=%s",
        document_name, page, limit,
    )

    if query_filter:

        _log.debug("Query filter: %s", query_filter)

    search_result = client.query_points(

        collection_name=COLLECTION_NAME,

        query=query_embedding,

        query_filter=query_filter,

        limit=limit

    )

    _log.info("Results found: %d", len(search_result.points))

    if search_result.points:

        _log.debug("First result payload: %s", search_result.points[0].payload)

    return search_result.points


# ==========================================================
# Page Texts (anchor lookup)
# ==========================================================

def get_page_texts(document_name: str, pages: list[int]) -> dict[int, str]:
    """Full text of the given pages of a document, from the stored parent
    chunks: {page: text}. Pages that aren't indexed are left out."""

    if not pages:
        return {}

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="document_name", match=MatchValue(value=document_name)),
            FieldCondition(key="page", match=MatchAny(any=list(pages))),
        ]),
        limit=len(pages) * 100,
        with_payload=["page", "parent_id", "parent_text"],
    )

    # A page longer than one parent chunk has several parents; children
    # repeat their parent's text, so keep each parent once.
    parents = {}

    for point in points:
        payload = point.payload or {}
        parents[(payload.get("page"), payload.get("parent_id"))] = payload.get("parent_text", "")

    texts = {}

    for (page, _), text in sorted(parents.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        texts[page] = (texts.get(page, "") + "\n" + text).strip()

    return texts
