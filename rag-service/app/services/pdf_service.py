import hashlib
import os
import time
import uuid

import fitz

from app.core.config import settings
from app.core.logger import logger


def count_pages(content: bytes) -> int:
    """Page count of an in-memory PDF; ValueError if it can't be opened."""

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            return document.page_count
    except Exception as exc:
        raise ValueError("Invalid PDF content.") from exc


# Cap the sanitized stem so the stored filename (stem + extension) can
# never approach OS path-length limits (Windows MAX_PATH=260 is the
# tightest common ceiling) regardless of how long the uploaded name
# was.
MAX_STEM_BYTES = 150

# When a stem has to be truncated, this many hex characters of a
# sha256 of the full original name are appended (see sanitize_filename)
# so two long names that happen to share a long common prefix don't
# collide on the same truncated stem, while the same long name always
# truncates to the same stored name (so a re-upload still matches).
NAME_HASH_LENGTH = 8
NAME_HASH_SEPARATOR = "-"

# The PDF spec allows bytes to precede the header, so a conforming
# reader looks for "%PDF-" anywhere in roughly the first KB rather than
# requiring it at byte 0.
PDF_HEADER_SEARCH_WINDOW = 1024


def validate_stored_filename(raw: str) -> str:
    """
    Validate that `raw` already looks like one of our own stored file
    names, without rewriting it in any way.

    Requires a bare basename -- no "/" or "\\" anywhere in it at all
    (unlike sanitize_filename, this never extracts a basename out of a
    larger path; it rejects the whole thing outright) -- ending in
    ".pdf" (case-insensitive), not "." or "..", with a non-empty stem.
    Raises ValueError otherwise.

    Used by the PDF preview route instead of sanitize_filename:
    preview looks up a name that was already chosen (and possibly
    already truncated/hashed) at upload time, so it must never
    re-derive or rewrite it. sanitize_filename's truncation threshold
    is a moving target (MAX_STEM_BYTES has already changed once across
    this branch) -- re-running it on a name already in, say, the
    151-213 byte range would silently map it to a *different* stored
    name and 404 a document that's actually sitting on disk under its
    original name. Path traversal is still stopped here, just by
    rejecting any separator outright instead of trying to extract a
    basename from one.
    """

    name = raw.strip()

    if "/" in name or "\\" in name:
        raise ValueError(f"Invalid filename: {raw!r}")

    if not name or name in (".", "..") or not name.lower().endswith(".pdf"):
        raise ValueError(f"Invalid filename: {raw!r}")

    if not name[:-4]:
        raise ValueError(f"Invalid filename: {raw!r}")

    return name


def sanitize_filename(raw: str) -> str:
    """
    Reduce an untrusted filename to a safe, NEW basename to store an
    upload under.

    Normalises Windows-style separators, keeps only the final path
    component, and strips whitespace. Raises ValueError if the result
    is empty, ".", "..", or doesn't end with ".pdf" (case-insensitive)
    -- this is what stops upload path traversal.

    The stem (everything but the ".pdf"/".PDF"/... extension) is capped
    at MAX_STEM_BYTES UTF-8 bytes so a very long name can never produce
    a path exceeding filesystem limits; truncation always lands on a
    whole-character boundary, never splitting a multi-byte character.

    When the stem is too long and has to be truncated, a short
    deterministic hash of the full original name is appended (cut
    stem + "-" + 8 hex chars of sha256(name)) so two different long
    names that share a long common prefix don't collide on the same
    stored name, while re-uploading the exact same long name always
    truncates to the same stored name.

    Only for choosing the stored name of a NEW upload -- looking up an
    EXISTING stored name (the PDF preview route) must use
    validate_stored_filename instead, since this truncates and that
    must not (see its docstring).
    """

    normalized = raw.replace("\\", "/")

    name = normalized.rsplit("/", 1)[-1].strip()

    if not name or name in (".", "..") or not name.lower().endswith(".pdf"):
        raise ValueError(f"Invalid filename: {raw!r}")

    stem, extension = name[:-4], name[-4:]

    if len(stem.encode("utf-8")) > MAX_STEM_BYTES:

        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:NAME_HASH_LENGTH]
        suffix = NAME_HASH_SEPARATOR + digest

        budget = MAX_STEM_BYTES - len(suffix.encode("utf-8"))
        stem = _truncate_utf8(stem, budget) + suffix

    if not stem:
        raise ValueError(f"Invalid filename: {raw!r}")

    return stem + extension


def _truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate `text` to at most `max_bytes` UTF-8 bytes without ever
    splitting a multi-byte character."""

    encoded = text.encode("utf-8")

    if len(encoded) <= max_bytes:
        return text

    truncated = encoded[:max_bytes]

    while truncated:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]

    return ""


def save_pdf_to_temp(content: bytes) -> str:
    """
    Write `content` to a short, uuid-named temp file under raw/, and
    validate that it looks like a PDF.

    The temp name is derived only from a uuid -- never from the
    caller's filename -- so an upload can never produce a path
    exceeding filesystem limits, regardless of how long the final name
    will be. Callers do all the fallible extraction/chunking/embedding/
    indexing work against this temp file *before* touching the
    document's real file (see document_service.ingest_document), and
    own removing it (e.g. in a `finally`) once they're done with it,
    whether or not it was ever published.

    Raises ValueError (and removes the temp file) if `content` doesn't
    look like a PDF -- the %PDF- marker must appear somewhere in the
    first PDF_HEADER_SEARCH_WINDOW bytes, as the spec allows.
    """

    upload_dir = settings.RAW_DATA_PATH

    os.makedirs(upload_dir, exist_ok=True)

    tmp_path = os.path.join(
        upload_dir,
        f".upload.{uuid.uuid4().hex}.tmp",
    )

    try:
        with open(tmp_path, "wb") as buffer:
            buffer.write(content)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    if b"%PDF-" not in content[:PDF_HEADER_SEARCH_WINDOW]:
        os.remove(tmp_path)
        raise ValueError("Invalid PDF content: no %PDF- header found.")

    return tmp_path


class PDFFileLockedError(Exception):
    """raw/<filename> is open in another program (Windows locks open
    files), so the new upload can't replace it."""

    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"{filename!r} is open in another program.")


# Antivirus scanners and indexers often hold a new file for a moment,
# so a few short retries cover those before blaming the user.
_PUBLISH_RETRIES = 5
_PUBLISH_RETRY_DELAY = 0.2


def publish_pdf(tmp_path: str, filename: str) -> str:
    """
    Atomically make `tmp_path` the current content of raw/<filename>,
    replacing whatever was there before (or creating it, on a first
    upload).

    A same-filesystem os.replace, so a reader of raw/<filename> -- the
    preview route, or this same function racing a concurrent request
    -- can only ever see the previous complete file or the new one,
    never a partially written one.
    """

    upload_dir = settings.RAW_DATA_PATH

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, filename)

    for attempt in range(_PUBLISH_RETRIES):
        try:
            os.replace(tmp_path, file_path)
            break
        except PermissionError:
            if attempt == _PUBLISH_RETRIES - 1:
                raise PDFFileLockedError(filename)
            time.sleep(_PUBLISH_RETRY_DELAY)

    logger.info(f"PDF published: {filename}")

    return file_path
