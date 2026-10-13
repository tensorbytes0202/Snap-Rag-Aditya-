import json
import logging
import os
import threading

from app.core.config import settings


logger = logging.getLogger(__name__)

# Guards every read-modify-write cycle against the registry file
# and the active-documents file so concurrent requests can't race
# each other into a torn write.
_lock = threading.Lock()


def _atomic_write(path, data):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    tmp_path = f"{path}.tmp"

    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)

        os.replace(tmp_path, path)

    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _load_registry():

    registry_file = settings.REGISTRY_FILE

    if not os.path.exists(registry_file):
        return {}

    with open(registry_file, "r") as f:
        return json.load(f)


def _save_registry(registry):
    _atomic_write(settings.REGISTRY_FILE, registry)


def _load_active_documents():

    active_file = settings.ACTIVE_DOCUMENTS_FILE

    if not os.path.exists(active_file):
        return {}

    with open(active_file, "r") as f:
        return json.load(f)


def _save_active_documents(active_documents):
    _atomic_write(settings.ACTIVE_DOCUMENTS_FILE, active_documents)


def is_document_indexed(document_name):

    with _lock:
        registry = _load_registry()

    return document_name in registry


def register_document(
    document_name,
    pages,
    chunks,
    uploaded_by: str | None = None,
    overview: list[str] | None = None,
):

    with _lock:

        registry = _load_registry()

        # `uploaded_by` is stored so a re-upload of an existing name can
        # be checked against its original owner (Ruling 20/23). It never
        # appears in get_documents()'s public shape, and neither does
        # `overview` (one gist line per page, see get_document_overview).
        registry[document_name] = {
            "pages": pages,
            "chunks": chunks,
            "uploaded_by": uploaded_by,
            "overview": overview or [],
        }

        _save_registry(registry)

        if uploaded_by:

            active_documents = _load_active_documents()
            active_documents[uploaded_by] = document_name
            _save_active_documents(active_documents)

    logger.info("Document registered: %s", document_name)


def get_documents(owner: str | None = None):
    """All documents, or with `owner` only the ones that user uploaded, so
    one user (e.g. a demo guest) never sees another user's PDFs."""

    with _lock:
        registry = _load_registry()

    documents = []

    for name, info in registry.items():

        if owner is not None and info.get("uploaded_by") != owner:
            continue

        documents.append({
            "name": name,
            "pages": info["pages"],
            "chunks": info["chunks"],
        })

    return documents


def get_document_overview(document_name) -> list[str]:
    """One gist line per summarized page ("Page 12: Verification result:
    ..."), or [] for documents without one (text-only or legacy)."""

    with _lock:
        registry = _load_registry()

    return registry.get(document_name, {}).get("overview", [])


def find_conflicting_name(document_name) -> str | None:
    """
    Return the existing registry name that differs from `document_name`
    only in letter case, or None if there is no such entry -- whether
    because nothing matches at all, or because the only match is an
    exact, byte-identical name (which is not a case conflict; that's
    handled by the ordinary owner-check re-upload path).
    """

    with _lock:
        registry = _load_registry()

    if document_name in registry:
        return None

    # .casefold(), not .lower(): "Straße".lower() == "straße" (the
    # eszett is untouched), so comparing against "STRASSE".lower() ==
    # "strasse" would miss the conflict. .casefold() folds both to
    # "strasse", and matches how the concurrent-upload processing
    # guard compares names (document_service._reserve_for_processing).
    folded = document_name.casefold()

    for existing_name in registry:
        if existing_name.casefold() == folded:
            return existing_name

    return None


def get_document_owner(document_name) -> str | None:
    """
    Return the `sub` of the user who (last) uploaded `document_name`, or
    None if it isn't registered or predates ownership tracking (a legacy
    entry with no `uploaded_by`).
    """

    with _lock:
        registry = _load_registry()

    return registry.get(document_name, {}).get("uploaded_by")


def set_active_document(user_id: str, document_name: str) -> None:

    with _lock:

        active_documents = _load_active_documents()
        active_documents[user_id] = document_name
        _save_active_documents(active_documents)

    logger.debug("Active document for %s set to %s", user_id, document_name)


def get_active_document(user_id: str) -> str | None:

    with _lock:
        active_documents = _load_active_documents()

    return active_documents.get(user_id)
