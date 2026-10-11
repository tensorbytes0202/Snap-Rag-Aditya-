import os
import pickle
import uuid

from app.core.config import settings

# NOTE: pickle is used here for the BM25 index cache. These files are
# generated and read back by this service only (never accepted from a
# client), so this is not a deserialization-of-untrusted-data concern.
# Pre-existing behavior; unchanged by this task beyond relocating the
# directory under STORAGE_DIR.


def stage_index(document_name, bm25, chunks) -> str:
    """
    Serialize `bm25`/`chunks` to a temp file under BM25_INDEX_DIR and
    return its path, WITHOUT touching document_name's current index.

    Callers finish the swap with commit_index (atomic os.replace into
    the final path) once every other fallible step of a re-index has
    also succeeded, or discard_staged_index to clean up on failure --
    this is what lets a re-upload leave the old BM25 index untouched
    when a later step (e.g. the Qdrant vector swap) fails.
    """

    index_dir = settings.BM25_INDEX_DIR

    os.makedirs(index_dir, exist_ok=True)

    tmp_path = os.path.join(
        index_dir,
        f".{uuid.uuid4().hex}.tmp",
    )

    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(
                {
                    "bm25": bm25,
                    "chunks": chunks,
                },
                f,
            )
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return tmp_path


def commit_index(tmp_path: str, document_name: str) -> None:
    """Atomically publish a staged index as the current one for
    `document_name` (a same-filesystem rename -- never a window with no
    index, or a partially written one)."""

    path = os.path.join(
        settings.BM25_INDEX_DIR,
        f"{document_name}.pkl",
    )

    os.replace(tmp_path, path)


def discard_staged_index(tmp_path: str) -> None:
    """Remove a staged index that will never be committed."""

    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)


def load_index(document_name):

    path = os.path.join(
        settings.BM25_INDEX_DIR,
        f"{document_name}.pkl"
    )

    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:

        return pickle.load(f)
