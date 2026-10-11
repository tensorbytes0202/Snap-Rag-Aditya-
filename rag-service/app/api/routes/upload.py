import logging
from pathlib import Path as _Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool

from app.core.security import get_current_user_id
from app.services.pdf_service import sanitize_filename, PDFFileLockedError
from app.services.document_service import (
    ingest_document,
    DocumentOwnershipError,
    DocumentBeingProcessedError,
    CaseInsensitiveNameConflictError,
    ScannedPDFError,
)

from app.services.document_registry import (
    set_active_document,
    get_active_document,
    get_documents,
)


logger = logging.getLogger(__name__)

router = APIRouter()

# Upload constraints
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB
ALLOWED_CONTENT_TYPES = {"application/pdf"}
ALLOWED_EXTENSIONS = {".pdf"}


# ==========================================================
# Upload Document
# ==========================================================

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Upload a PDF document and process it
    Pipeline:
       Upload → Validate → Extract → Embed → Qdrant
    """
    # ------------------------------------------------------
    # Validate file extension
    # ------------------------------------------------------
    ext = _Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only PDF files are accepted.",
        )

    # ------------------------------------------------------
    # Validate MIME type reported by client
    # (not fully trusted, but adds a basic layer)
    # ------------------------------------------------------
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type '{file.content_type}'. Only application/pdf is accepted.",
        )

    # ------------------------------------------------------
    # Validate file size (read into memory; used directly below,
    # no need to seek back since we no longer re-read file.file)
    # ------------------------------------------------------
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
        )

    # ------------------------------------------------------
    # Sanitize filename (path traversal protection)
    # ------------------------------------------------------
    try:
        safe_name = sanitize_filename(file.filename or "")
    except ValueError:
        logger.warning("Rejected unsafe upload filename: %s", file.filename)
        raise HTTPException(status_code=400, detail="Invalid filename.")

    logger.info("Upload received: %s (user=%s)", safe_name, user_id)

    # ------------------------------------------------------
    # Ingest the document off the event loop -- the disk write,
    # extraction, chunking, embedding and indexing are all
    # blocking/CPU-bound and would otherwise stall every other
    # request. ingest_document also reserves `safe_name` against
    # concurrent processing and enforces the per-owner overwrite
    # rule before touching the file on disk.
    # ------------------------------------------------------

    try:
        result = await run_in_threadpool(
            ingest_document,
            content=content,
            filename=safe_name,
            uploaded_by=user_id,
        )
    except DocumentBeingProcessedError:
        raise HTTPException(
            status_code=409,
            detail="This document is already being processed.",
        )
    except CaseInsensitiveNameConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"A document with this name already exists: {exc.existing_name}",
        )
    except DocumentOwnershipError:
        raise HTTPException(
            status_code=409,
            detail="A document with this name was uploaded by another user.",
        )
    except PDFFileLockedError:
        raise HTTPException(
            status_code=409,
            detail=(
                "The stored copy of this PDF is open in another program. "
                "Close it and upload again."
            ),
        )
    except ScannedPDFError:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text could be extracted from this PDF, even with OCR. "
                "It may be blank, corrupted, or contain no legible text."
            ),
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid PDF file.")

    return result


# ==========================================================
# List Documents
# ==========================================================

@router.get("/documents")
def list_documents(
    user_id: str = Depends(get_current_user_id),
):
    """
    Return all documents registered in the RAG system, plus the
    caller's own active document.
    """

    return {
        "documents": get_documents(),
        "active_document": get_active_document(user_id),
    }


# ==========================================================
# Set Active Document
# ==========================================================

@router.post(
    "/documents/active/{document_name}"
)
def select_document(
    document_name: str,
    user_id: str = Depends(get_current_user_id),
):
    """
    Set a document as the caller's currently active document.
    """

    if not any(
        doc["name"] == document_name
        for doc in get_documents()
    ):
        raise HTTPException(status_code=404, detail="Document not found.")

    set_active_document(
        user_id,
        document_name,
    )

    return {
        "message": "Active document updated.",
        "active_document": document_name,
    }


# ==========================================================
# Get Active Document
# ==========================================================

@router.get("/documents/active")
def active_document(
    user_id: str = Depends(get_current_user_id),
):
    """
    Return the caller's currently active document.
    """

    return {
        "active_document": get_active_document(user_id)
    }
