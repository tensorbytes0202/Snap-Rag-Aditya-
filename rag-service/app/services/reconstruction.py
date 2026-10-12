"""Verified reconstruction of snapshot memory (the paper's core method).

    reconstruct  snapshot (+ the document pages it is anchored to)
                 -> LLM -> short, self-contained sentences
    verify       each sentence, as a cascade:
                   1. do the snapshot facts or the document passages state it?
                   2. if not, does the archive (the verbatim turns the
                      snapshot replaced) state it?
                 sentences no source states are dropped, so only verified
                 content can reach a prompt.

The modes are the ablation ladder of the paper:
    "snapshot"  reconstruct from the snapshot alone, no checks
    "evidence"  + document passages, no checks
    "verify"    + step 1, unsupported sentences dropped
    "archive"   + step 2 for what step 1 could not confirm (full method)

The archive is the chat log itself: append-only and never rewritten, cut to
the snapshot's turn range.
"""

import logging

from langsmith import traceable

from app.core.config import settings
from app.prompts.reconstruction_prompt import (
    build_judge_prompt,
    build_reconstruct_prompt,
    render_archive,
    render_evidence,
    render_snapshot_for_prompt,
)
from app.repositories.vector_repository import get_page_texts
from app.services.llm_provider import chat
from app.services.session_store import get_turns
from app.utils.json_utils import extract_json


_log = logging.getLogger(__name__)

MODES = ("snapshot", "evidence", "verify", "archive")


def _model() -> str | None:

    return settings.RECONSTRUCTION_MODEL or None


# ==========================================================
# Evidence: the document pages a snapshot is anchored to
# ==========================================================

def load_evidence(snapshot: dict) -> dict:
    """{(document, page): text} for the snapshot's anchor pages, in the
    order its facts cite them, capped at EVIDENCE_MAX_CHARS."""

    wanted = []

    for fact in snapshot["facets"]["facts"]:
        for page in fact.get("pages") or []:
            if fact.get("document") and (fact["document"], page) not in wanted:
                wanted.append((fact["document"], page))

    texts = {}

    for document in dict.fromkeys(document for document, _ in wanted):
        pages = get_page_texts(document, [page for doc, page in wanted if doc == document])
        texts.update({(document, page): text for page, text in pages.items()})

    evidence, used = {}, 0

    for key in wanted:

        text = texts.get(key)

        if not text:
            continue

        if evidence and used + len(text) > settings.EVIDENCE_MAX_CHARS:
            break

        evidence[key] = text
        used += len(text)

    return evidence


# ==========================================================
# Reconstruct
# ==========================================================

def reconstruct(snapshot: dict, evidence: dict, question: str | None = None) -> list[str]:

    prompt = build_reconstruct_prompt(snapshot, evidence, question)

    data = extract_json(chat([{"role": "user", "content": prompt}], model=_model())["message"]["content"])

    sentences = data.get("sentences") if isinstance(data, dict) else None

    if not isinstance(sentences, list):
        raise ValueError("reconstruction did not return a sentence list")

    return [s.strip() for s in sentences if isinstance(s, str) and s.strip()]


# ==========================================================
# Judge
# ==========================================================

def judge(sentences: list[str], sources: dict[str, str], model: str | None = None) -> list[str] | None:
    """One label per sentence: the first source that states it, or "none".
    None if the judge call itself failed. A sentence the judge skips counts
    as "none", so a sloppy reply can only drop content, never let it through."""

    if not sentences:
        return []

    try:
        reply = chat([{"role": "user", "content": build_judge_prompt(sentences, sources)}], model=model)
        data = extract_json(reply["message"]["content"])
    except Exception:
        _log.warning("Judge call failed", exc_info=True)
        return None

    checks = data.get("checks") if isinstance(data, dict) else None

    if not isinstance(checks, list):
        return None

    labels = ["none"] * len(sentences)
    allowed = set(sources) | {"none"}

    for check in checks:

        if not isinstance(check, dict) or check.get("source") not in allowed:
            continue

        try:
            index = int(check.get("i"))
        except (TypeError, ValueError):
            continue

        if 1 <= index <= len(sentences):
            labels[index - 1] = check["source"]

    return labels


# ==========================================================
# Recall: reconstruct + verify
# ==========================================================

@traceable(name="recall_memory", run_type="chain")
def recall(user_id: str, chat_id: str, snapshot: dict, question: str | None = None, mode: str = "archive") -> dict:
    """Rebuild what `snapshot` remembers about `question` and, in the
    verified modes, keep only sentences a source confirms."""

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    evidence = {} if mode == "snapshot" else load_evidence(snapshot)

    sentences = reconstruct(snapshot, evidence, question)

    result = {
        "mode": mode,
        "snapshot": snapshot["state_hash"],
        "question": question,
        "evidence_pages": [f"{document} p{page}" for document, page in evidence],
        "verification_failed": False,
    }

    if mode in ("snapshot", "evidence"):
        result["kept"] = [{"text": s, "source": None} for s in sentences]
        result["dropped"] = []
        return result

    labels = judge(sentences, {
        "snapshot": render_snapshot_for_prompt(snapshot),
        "document": render_evidence(evidence) or "(none)",
    }, model=_model())

    if labels is None:
        # Without a working check nothing counts as verified.
        labels, result["verification_failed"] = ["none"] * len(sentences), True

    doubtful = [i for i, label in enumerate(labels) if label == "none"]

    if mode == "archive" and doubtful and not result["verification_failed"]:

        archive = [
            turn for turn in get_turns(user_id, chat_id)
            if snapshot["from_turn"] <= turn["turn"] <= snapshot["to_turn"]
        ]

        rescued = judge([sentences[i] for i in doubtful], {"archive": render_archive(archive)}, model=_model())

        if rescued is None:
            result["verification_failed"] = True
        else:
            for i, label in zip(doubtful, rescued):
                labels[i] = label

    result["kept"] = [{"text": s, "source": label} for s, label in zip(sentences, labels) if label != "none"]
    result["dropped"] = [s for s, label in zip(sentences, labels) if label == "none"]

    return result
