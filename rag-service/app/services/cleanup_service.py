import logging
import os
import time

from app.core.config import settings


_log = logging.getLogger(__name__)

# A file this old is assumed to be left behind by a killed/crashed
# process rather than a genuinely in-flight upload -- uploads finish
# (or clean up after themselves) in well under this window.
STALE_TEMP_FILE_AGE_SECONDS = 10 * 60


def _remove_stale_temp_files(directory: str, prefix: str, suffix: str, max_age_seconds: float) -> int:
    """
    Remove files directly under `directory` whose name starts with
    `prefix` and ends with `suffix` and whose mtime is older than
    `max_age_seconds`. Returns the count removed. Never raises for a
    missing directory, and skips (rather than crashes on) any single
    file it can't stat or remove -- another process may be touching it
    at the same moment.
    """

    if not os.path.isdir(directory):
        return 0

    cutoff = time.time() - max_age_seconds
    removed = 0

    with os.scandir(directory) as entries:

        for entry in entries:

            if not entry.name.startswith(prefix) or not entry.name.endswith(suffix):
                continue

            try:
                if not entry.is_file():
                    continue

                if entry.stat().st_mtime >= cutoff:
                    continue

                os.remove(entry.path)
                removed += 1

            except OSError:
                continue

    return removed


def sweep_stale_temp_files(max_age_seconds: float = STALE_TEMP_FILE_AGE_SECONDS) -> int:
    """
    Remove temp files a killed/crashed process left behind:
      - `.upload.<uuid>.tmp` in the raw PDF directory
        (pdf_service.save_pdf_to_temp)
      - `.<uuid>.tmp` in the BM25 index directory
        (bm25_index.stage_index)

    Only files older than `max_age_seconds` are removed, so a temp
    file from an upload that's still genuinely in flight is never
    touched. Meant to be called once, at startup (see
    app.core.lifespan). Logs only the count removed, at INFO --
    never a file name or any file content.
    """

    removed = 0

    removed += _remove_stale_temp_files(
        settings.RAW_DATA_PATH, ".upload.", ".tmp", max_age_seconds,
    )

    removed += _remove_stale_temp_files(
        settings.BM25_INDEX_DIR, ".", ".tmp", max_age_seconds,
    )

    _log.info("Startup sweep removed %d stale temp file(s)", removed)

    return removed
