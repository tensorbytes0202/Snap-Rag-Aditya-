import logging
import os
import threading
from contextlib import contextmanager

from app.services.pdf_extractor import extract_pdf_data
from app.services.chunker import create_parent_child_chunks
from app.services.embedding_service import generate_embeddings

from app.services.document_registry import (
    is_document_indexed,
    register_document,
    get_document_owner,
    find_conflicting_name,
)

from app.services.pdf_service import save_pdf_to_temp, publish_pdf
from app.cache.cache_service import clear_all_cache

from app.services.bm_25_service import (
    stage_bm25_index,
    commit_bm25_index,
    discard_bm25_index,
)

from app.repositories.vector_repository import (
    create_collection,
    store_embeddings,
)


_log = logging.getLogger(__name__)

# Guards against two uploads of the same filename running the
# extract/embed/store/BM25 pipeline concurrently -- without this, their
# store_embeddings calls can interleave and leave duplicate vectors
# behind.
_processing_lock = threading.Lock()
_processing_names: set[str] = set()

# Budget for the per-page gist lines sent with every query on this
# document (see query_runner.Retrieval.top_context): roughly 1.5k tokens.


class DocumentOwnershipError(Exception):
    """A different user already owns a document registered under this name."""

    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"Document {filename!r} was uploaded by another user.")


class DocumentBeingProcessedError(Exception):
    """This document name is already being ingested by another request."""

    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"Document {filename!r} is already being processed.")


class CaseInsensitiveNameConflictError(Exception):
    """A document whose name differs only in letter case already exists."""

    def __init__(self, existing_name: str):
        self.existing_name = existing_name
        super().__init__(f"A document with this name already exists: {existing_name}")


class ScannedPDFError(Exception):
    """No text could be extracted, even after the OCR fallback -- the PDF
    is likely blank, corrupted, or an image OCR can't read (e.g. a photo
    with no legible text)."""

    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(
            "No text could be extracted from this PDF. "
            "Scanned or image-only PDFs are not supported."
        )


@contextmanager
def _reserve_for_processing(filename: str):

    # Keyed on the casefolded name so two uploads that differ only in
    # letter case (e.g. "Report.pdf" and "report.pdf") are serialized
    # against each other too, not just byte-identical names -- the
    # second one gets DocumentBeingProcessedError while the first is
    # still in flight, instead of both racing the case-insensitive
    # conflict check below.
    key = filename.casefold()

    with _processing_lock:

        if key in _processing_names:
            raise DocumentBeingProcessedError(filename)

        _processing_names.add(key)

    try:
        yield

    finally:

        with _processing_lock:
            _processing_names.discard(key)


# ==========================================================
# Ingest Document
# ==========================================================
#
# Entry point for an upload: reserves the filename against concurrent
# processing, enforces the per-owner overwrite rule and the
# case-insensitive name-conflict rule, then runs the full,
# all-or-nothing ingestion pipeline. Meant to be called via
# run_in_threadpool -- everything here is blocking, CPU/IO-bound work.

def ingest_document(
    content: bytes,
    filename: str,
    uploaded_by: str | None = None,
):

    with _reserve_for_processing(filename):

        pre_existing = is_document_indexed(filename)

        if pre_existing:

            owner = get_document_owner(filename)

            if owner and uploaded_by and owner != uploaded_by:
                raise DocumentOwnershipError(filename)

        else:

            conflicting_name = find_conflicting_name(filename)

            if conflicting_name is not None:
                raise CaseInsensitiveNameConflictError(conflicting_name)

        # All the fallible work -- extraction, chunking, embedding and
        # indexing -- happens against this temp copy. The document's
        # existing file, vectors, BM25 index and registry entry are
        # never touched unless every one of those steps succeeds.
        tmp_path = save_pdf_to_temp(content)

        try:

            result = process_document(
                file_path=tmp_path,
                filename=filename,
                uploaded_by=uploaded_by,
                pre_existing=pre_existing,
            )

            # Publish before registering: a reader must never see a
            # registry entry for a document whose raw file doesn't
            # exist or is stale. This os.replace is a same-filesystem
            # rename, so raw/<filename> is never missing or partially
            # written either way.
            publish_pdf(tmp_path, filename)

            # Only after the file is actually in place do we record
            # it. If register_document itself then fails (a local
            # JSON write -- much less likely than anything above), the
            # new file/vectors/BM25 index are already live but the
            # registry still shows the old (or no) entry; the next
            # upload of this name is simply treated as unregistered
            # and safely redoes the whole pipeline, which is a far
            # safer failure mode than the registry pointing at a file
            # that was never published.
            register_document(
                filename,
                result["pages"],
                result["total_chunks"],
                uploaded_by=uploaded_by,
            )

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if pre_existing:

            # Re-indexing an existing name means any answer cached for
            # it may now describe stale content -- see clear_all_cache
            # for why this can't be targeted to just this document.
            clear_all_cache()

        return result


# ==========================================================
# Process Document
# ==========================================================
#
# Runs every fallible step (extraction, chunking, embedding, indexing)
# against `file_path` (the temp copy) and only registers the result
# once the index swap has fully succeeded.

def process_document(
    file_path: str,
    filename: str,
    uploaded_by: str | None = None,
    pre_existing: bool = False,
):

    # ======================================================
    # Extract PDF
    # ======================================================

    _log.info("Processing document: %s", filename)

    pdf_data = extract_pdf_data(
        file_path
    )

    _log.info("Total pages: %d", pdf_data["total_pages"])

    total_text = "".join(
        page["text"] for page in pdf_data["pages"]
    ).strip()

    if not total_text:
        raise ScannedPDFError(filename)

    # ======================================================
    # Create Parent / Child Chunks
    # ======================================================

    chunks = []
    parents = []

    for page in pdf_data["pages"]:

        page_text = page["text"]

        parent_documents = create_parent_child_chunks(
            page_text
        )

        for parent in parent_documents:

            parent_unique_id = (
                f"{filename}"
                f"_page_{page['page_number']}"
                f"_parent_{parent['parent_id']}"
            )

            parents.append(
                {
                    "parent_id": parent_unique_id,
                    "parent_text": parent["parent_text"],
                    "page": page["page_number"],
                    "document": filename,
                }
            )

            for child in parent["children"]:

                chunks.append(
                    {
                        "text": child["text"],
                        "page": page["page_number"],
                        "document": filename,
                        "parent_id": parent_unique_id,
                        "parent_text": parent["parent_text"],
                        "child_id": child["child_id"],
                    }
                )

    _log.info("Total chunks: %d", len(chunks))

    # ======================================================
    # Generate Embeddings
    # ======================================================

    embeddings = generate_embeddings(
        [
            chunk["text"]
            for chunk in chunks
        ]
    )

    _log.info("Total embeddings: %d", len(embeddings))

    # ======================================================
    # Swap the index: stage the new BM25 index, write the new
    # vectors (removing the old ones only once the new ones are
    # safely in), then publish the BM25 index. If storing the
    # vectors fails, the staged BM25 index is discarded and the old
    # BM25 index/vectors are left exactly as they were.
    # ======================================================

    create_collection()

    bm25_tmp_path = stage_bm25_index(filename, chunks)

    try:
        store_embeddings(
            chunks,
            embeddings,
            filename,
            pre_existing=pre_existing,
        )
        commit_bm25_index(bm25_tmp_path, filename)
    except Exception:
        discard_bm25_index(bm25_tmp_path)
        raise

    # Registration happens in ingest_document, after the raw file is
    # published -- see the comment there for why.

    # ======================================================
    # Result
    # ======================================================

    return {
        "filename": filename,
        "pages": pdf_data["total_pages"],
        "total_chunks": len(chunks),
        "vectors_stored": len(embeddings),
        "already_indexed": False,
    }
