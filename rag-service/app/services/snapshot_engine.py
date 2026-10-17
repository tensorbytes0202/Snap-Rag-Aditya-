"""Snapshot engine: compresses a window of chat turns into an immutable snapshot.

The LLM extracts the four facets (facts, entities, decisions, open
questions). Code then attaches the document/page anchors each source turn
cited, links the snapshot to the previous one (prev_hash), hashes it
(state_hash), writes it once and indexes it for recall.

after_turn() runs after every recorded chat turn: the context monitor
measures the window, and a snapshot is taken when it triggers one. The
next turn therefore starts with the memory already compressed, which is
the "context check before the query" step of the design.
"""

import hashlib
import json
import logging
import threading
from datetime import datetime, timezone

from langsmith import traceable

from app.api.dependencies.embedding import get_embedding_model
from app.core.config import settings
from app.prompts.snapshot_prompt import build_snapshot_prompt
from app.services.context_monitor import CHARS_PER_TOKEN, decide, measure, turn_tokens
from app.services.llm_provider import chat
from app.services.memory_state import update_state
from app.services.session_store import get_turns
from app.services.snapshot_store import (
    append_cqm,
    index_snapshot,
    list_snapshots,
    read_cqm,
    save_snapshot,
)
from app.utils.json_utils import extract_json


_log = logging.getLogger(__name__)

# Facets the LLM writes; the others are derived by code. Only these are shown
# to the extractor as examples.
LLM_FACETS = ("facts", "user_facts", "entities", "decisions", "open_questions")

ASKED_CHARS = 160


# ==========================================================
# Facet cleanup and anchors
# ==========================================================

def _as_list(value) -> list:
    """LLM output is untrusted: a facet or turn list may come back as a
    single value or something else entirely."""

    if isinstance(value, list):
        return value

    return [] if value is None else [value]


def _as_turn(value, valid: set) -> int | None:

    try:
        turn = int(value)
    except (TypeError, ValueError):
        return None

    return turn if turn in valid else None


def _anchors(turn: dict) -> tuple[str | None, list[int]]:
    """(document, cited pages) of a turn."""

    pages = sorted({
        citation["page"] for citation in turn.get("citations") or []
        if isinstance(citation.get("page"), int)
    })

    return turn.get("document"), pages


def _clean_facets(data: dict, window: list[dict]) -> dict:
    """Keep only well-formed items that point at turns of this window, and
    attach anchors from those turns (never from the LLM)."""

    by_turn = {turn["turn"]: turn for turn in window}
    valid = set(by_turn)

    facts, seen = [], set()

    for item in _as_list(data.get("facts")):

        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue

        turn = _as_turn(item.get("turn"), valid)
        text = item["text"].strip()

        if turn is None or not text or text.lower() in seen:
            continue

        seen.add(text.lower())
        document, pages = _anchors(by_turn[turn])

        facts.append({
            "text": text,
            "turn": turn,
            "document": document,
            "pages": pages,
            # True/False from answer verification; None if it never ran.
            "verified": by_turn[turn].get("verified"),
        })

    user_facts, seen_user = [], set()

    for item in _as_list(data.get("user_facts")):

        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue

        turn = _as_turn(item.get("turn"), valid)
        text = item["text"].strip()

        if turn is None or not text or text.lower() in seen_user:
            continue

        seen_user.add(text.lower())
        user_facts.append({"text": text, "turn": turn})

    entities = []

    for item in _as_list(data.get("entities")):

        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue

        turns = sorted({
            turn for turn in (_as_turn(v, valid) for v in _as_list(item.get("turns")))
            if turn is not None
        })

        if not turns or not item["name"].strip():
            continue

        entities.append({
            "name": item["name"].strip(),
            "turns": turns,
            "documents": sorted({by_turn[t].get("document") for t in turns if by_turn[t].get("document")}),
            "pages": sorted({page for t in turns for page in _anchors(by_turn[t])[1]}),
        })

    decisions = []

    for item in _as_list(data.get("decisions")):

        if not isinstance(item, dict) or not isinstance(item.get("topic"), str):
            continue

        turn = _as_turn(item.get("turn"), valid)

        if turn is None:
            continue

        decisions.append({
            "topic": item["topic"].strip(),
            "status": "closed" if item.get("status") == "closed" else "open",
            "turn": turn,
        })

    open_questions = [
        q.strip() for q in _as_list(data.get("open_questions"))
        if isinstance(q, str) and q.strip()
    ]

    # Written by code from the turns, not by the LLM: what was asked at which
    # turn, so "which case did I ask about at turn 5?" has a source.
    asked = [
        {"turn": turn["turn"], "text": " ".join(turn["question"].split())[:ASKED_CHARS]}
        for turn in window
    ]

    return {
        "facts": facts,
        "user_facts": user_facts,
        "entities": entities,
        "decisions": decisions,
        "open_questions": open_questions,
        "asked": asked,
    }


def render_snapshot(snapshot: dict) -> str:
    """Plain-text form of a snapshot, used for its embedding."""

    facets = snapshot["facets"]

    lines = [fact["text"] for fact in facets["facts"]]
    lines += [f"User said: {fact['text']}" for fact in facets.get("user_facts", [])]
    lines += [f"Entity: {entity['name']}" for entity in facets["entities"]]
    lines += [f"Decision ({d['status']}): {d['topic']}" for d in facets["decisions"]]
    lines += [f"Open question: {q}" for q in facets["open_questions"]]
    lines += [f"Asked at turn {a['turn']}: {a['text']}" for a in facets.get("asked", [])]

    return "\n".join(lines)


# ==========================================================
# Create Snapshot
# ==========================================================

@traceable(name="create_snapshot", run_type="chain")
def create_snapshot(user_id: str, chat_id: str, window: list[dict], previous: list[dict], meta: dict) -> dict:
    """Compress `window` into the next snapshot of the chat and store it.
    Raises if extraction fails (nothing is written then)."""

    exemplars = previous[-settings.SNAPSHOT_EXEMPLARS:] if settings.SNAPSHOT_EXEMPLARS > 0 else []

    prompt = build_snapshot_prompt(
        window,
        [{name: snapshot["facets"].get(name, []) for name in LLM_FACETS} for snapshot in exemplars],
    )

    data = extract_json(chat([{"role": "user", "content": prompt}])["message"]["content"])

    if not isinstance(data, dict):
        raise ValueError("snapshot extraction did not return a JSON object")

    body = {
        "seq": len(previous) + 1,
        "from_turn": window[0]["turn"],
        "to_turn": window[-1]["turn"],
        "prev_hash": previous[-1]["state_hash"] if previous else None,
        "facets": _clean_facets(data, window),
    }

    # Hash of the canonical content, including prev_hash, so each snapshot
    # also pins the whole chain before it.
    state_hash = hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()

    snapshot = {
        **body,
        "state_hash": state_hash,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exemplars_used": [e["state_hash"] for e in exemplars],
        "meta": meta,
    }

    text = render_snapshot(snapshot)
    source_tokens = sum(turn_tokens(turn) for turn in window)
    snapshot_tokens = max(1, len(text) // CHARS_PER_TOKEN)

    snapshot["meta"] = {
        **meta,
        "source_tokens": source_tokens,
        "snapshot_tokens": snapshot_tokens,
        "compression": round(source_tokens / snapshot_tokens, 2),
    }

    save_snapshot(user_id, chat_id, snapshot)

    try:
        vector = get_embedding_model().encode(text, normalize_embeddings=True).tolist()
        index_snapshot(user_id, chat_id, snapshot, text, vector)
    except Exception:
        _log.warning("Snapshot %s saved but not indexed", state_hash[:12], exc_info=True)

    _log.info(
        "Snapshot %d of chat %s: turns %d-%d, %d facts, %.1fx compression",
        snapshot["seq"], chat_id, body["from_turn"], body["to_turn"],
        len(body["facets"]["facts"]), snapshot["meta"]["compression"],
    )

    return snapshot


# ==========================================================
# After each turn: measure, maybe snapshot
# ==========================================================

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _chat_lock(chat_id: str) -> threading.Lock:

    with _locks_guard:
        return _locks.setdefault(chat_id, threading.Lock())


def after_turn(user_id: str, chat_id: str) -> None:
    """Background task after a chat turn is recorded. Never raises: memory
    upkeep must not affect answering."""

    try:
        with _chat_lock(chat_id):
            _after_turn(user_id, chat_id)
    except Exception:
        _log.exception("Memory update failed for chat %s", chat_id)


def _after_turn(user_id: str, chat_id: str) -> None:

    turns = get_turns(user_id, chat_id)

    # Nothing new: no turn yet, or the latest one was already measured
    # (e.g. a stream that ended in an error recorded no turn).
    if not turns or turns[-1]["turn"] in {r["turn"] for r in read_cqm(user_id, chat_id)}:
        return

    previous = list_snapshots(user_id, chat_id)
    covered = previous[-1]["to_turn"] if previous else 0
    window = [turn for turn in turns if turn["turn"] > covered]

    signals = measure(window)
    trigger = decide(signals, len(window))

    record = {
        "turn": turns[-1]["turn"],
        "window": [window[0]["turn"], window[-1]["turn"]],
        **signals,
        "trigger": trigger,
        "snapshot": None,
    }

    if trigger:
        try:
            snapshot = create_snapshot(user_id, chat_id, window, previous, {
                "trigger": trigger,
                "cis": signals["cis"],
                "headroom": signals["headroom"],
            })
            record["snapshot"] = snapshot["state_hash"]
        except Exception as exc:
            # The window keeps growing, so the next turn triggers again.
            _log.warning("Snapshot failed for chat %s: %s", chat_id, exc)
            record["snapshot_error"] = str(exc)[:200]

    append_cqm(user_id, chat_id, record)

    # The conversation state and hand-off follow the snapshots. It catches up
    # on every snapshot it has not seen, so a failure here is retried by the
    # next call and never loses a statement.
    if settings.MEMORY_STATE_ENABLED:
        try:
            update_state(user_id, chat_id)
        except Exception:
            _log.warning("State update failed for chat %s", chat_id, exc_info=True)
