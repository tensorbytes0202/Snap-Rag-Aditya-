"""Conversation state (the memory updater) and the hand-off document.

What the user stated or decided ("my deadline is 15 November", "I will skip
sections 32 and 33") is tiny, always relevant and found in no document. It is
therefore kept apart from the document facts, as a consolidated state that is
shown on every turn instead of competing with them for a snapshot-recall slot.

The state is built from the chat's snapshots, oldest first:

    1. candidates  the snapshot's decisions, user facts and open questions
    2. verify      each must be stated by the user's own messages in the
                   snapshot's turns (the chat record); the rest are rejected.
                   If the check itself fails, nothing is added and the snapshot
                   stays pending, so the next update retries it.
    3. consolidate an LLM names which earlier items a new one replaces (the
                   user changed their mind) and which open questions it
                   answers. The LLM only points at items; their text is never
                   rewritten, so it cannot introduce a fact.

Every update appends a version to state.jsonl (versions are never rewritten).
The newest version is the current state.

The hand-off document is built on request: the state plus where the conversation stood (last
question, topics, entities). It is shown to the model on the first turn of a
resumed session, and returned by GET /chats/{id}/handoff.
"""

import logging
import re
import time
from collections import Counter
from datetime import datetime, timezone

from langsmith import traceable

from app.core.config import settings
from app.services.context_monitor import CHARS_PER_TOKEN
from app.services.llm_provider import chat
from app.services.reconstruction import judge
from app.services.session_store import get_turns
from app.services.snapshot_store import (
    append_state_version,
    list_snapshots,
    read_state_versions,
)
from app.utils.json_utils import extract_json


_log = logging.getLogger(__name__)

STATE_HEADER = (
    "CONVERSATION STATE (what the user stated or decided earlier in this conversation, "
    "checked against the conversation record; if two items conflict, the later turn wins):"
)

_SUPERSEDE_PROMPT = """Below are items remembered from a conversation. OLD items were saved earlier;
NEW items were just added.

A NEW item REPLACES an OLD item when the user changed their mind or gave a different value for
the same thing (a new deadline, a different choice of scope). A NEW item ANSWERS an OPEN QUESTION
when it settles it. Items about different things replace nothing.

OLD:
{old}

NEW:
{new}

Return ONLY JSON: {{"replaced": [{{"old": <old number>, "by": <new number>}}], "answered": [<old open-question numbers>]}}
Use empty lists when nothing is replaced or answered."""


def _norm(text: str) -> str:

    return re.sub(r"\s+", " ", text.lower()).strip(" .")


# ==========================================================
# Items
# ==========================================================

def candidates(snapshot: dict) -> list[dict]:
    """The state items a snapshot proposes."""

    facets = snapshot["facets"]
    source = snapshot["state_hash"]

    items = [
        {"kind": "decision", "text": d["topic"], "turn": d["turn"], "status": d["status"], "snapshot": source}
        for d in facets.get("decisions", []) if d.get("topic")
    ]

    items += [
        {"kind": "user_fact", "text": u["text"], "turn": u["turn"], "snapshot": source}
        for u in facets.get("user_facts", [])
    ]

    items += [
        {"kind": "open_question", "text": q, "turn": snapshot["to_turn"], "snapshot": source}
        for q in facets.get("open_questions", [])
    ]

    return items


def render_item(item: dict) -> str:

    if item["kind"] == "decision":
        undecided = " (still undecided)" if item.get("status") == "open" else ""
        return f"User decision (turn {item['turn']}){undecided}: {item['text']}"

    if item["kind"] == "user_fact":
        return f"User stated (turn {item['turn']}): {item['text']}"

    return f"Open question (turn {item['turn']}): {item['text']}"


def _claim(item: dict) -> str:
    """The item as one sentence the check can test against the user's words."""

    if item["kind"] == "decision":
        return f"The user decided or committed to: {item['text']}"

    if item["kind"] == "user_fact":
        return item["text"]

    return f"The user asked: {item['text']}"


# ==========================================================
# Build the state from snapshots
# ==========================================================

def _verify(new: list[dict], snapshot: dict, turns: list[dict]) -> tuple[list[dict], list[dict]] | None:
    """(verified, rejected) items, or None if the check could not run."""

    if not new:
        return [], []

    window = [t for t in turns if snapshot["from_turn"] <= t["turn"] <= snapshot["to_turn"]]
    messages = "\n\n".join(f"Turn {t['turn']} (user): {t['question']}" for t in window) or "(empty)"

    labels = judge([_claim(item) for item in new], {"user_messages": messages})

    if labels is None:
        return None

    verified = [item for item, label in zip(new, labels) if label != "none"]
    rejected = [{"text": item["text"], "kind": item["kind"], "turn": item["turn"]}
                for item, label in zip(new, labels) if label == "none"]

    return verified, rejected


@traceable(name="consolidate_state", run_type="chain")
def _consolidate(old: list[dict], new: list[dict]) -> tuple[set[tuple[int, int]], set[int]]:
    """({(old index, new index)} replaced, {old open-question index} answered).
    A failed call changes nothing."""

    if not old or not new:
        return set(), set()

    prompt = _SUPERSEDE_PROMPT.format(
        old="\n".join(f"[{i}] {render_item(item)}" for i, item in enumerate(old, 1)),
        new="\n".join(f"[{i}] {render_item(item)}" for i, item in enumerate(new, 1)),
    )

    try:
        data = extract_json(chat([{"role": "user", "content": prompt}])["message"]["content"])
    except Exception:
        _log.warning("State consolidation call failed; keeping every item", exc_info=True)
        return set(), set()

    if not isinstance(data, dict):
        return set(), set()

    def valid(value, size):
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number - 1 if 1 <= number <= size else None

    replaced, answered = set(), set()

    for pair in data.get("replaced") or []:
        if isinstance(pair, dict):
            old_index = valid(pair.get("old"), len(old))
            new_index = valid(pair.get("by"), len(new))
            if old_index is not None and new_index is not None:
                replaced.add((old_index, new_index))

    for value in data.get("answered") or []:
        index = valid(value, len(old))
        if index is not None and old[index]["kind"] == "open_question":
            answered.add(index)

    return replaced, answered


def current_state(user_id: str, chat_id: str) -> list[dict]:
    """The newest version's items, oldest turn first."""

    versions = read_state_versions(user_id, chat_id)

    return versions[-1]["items"] if versions else []


@traceable(name="update_state", run_type="chain")
def update_state(user_id: str, chat_id: str) -> dict | None:
    """Fold every snapshot the state has not seen into it and save a new
    version. Returns that version, or None if nothing changed."""

    snapshots = list_snapshots(user_id, chat_id)
    versions = read_state_versions(user_id, chat_id)

    covered = list(versions[-1]["covered"]) if versions else []
    items = [dict(item) for item in versions[-1]["items"]] if versions else []

    pending = [s for s in snapshots if s["state_hash"] not in covered]

    if not pending:
        return None

    turns = get_turns(user_id, chat_id)
    superseded, rejected = [], []

    for snapshot in pending:

        known = {_norm(item["text"]) for item in items}
        proposed = [c for c in candidates(snapshot) if _norm(c["text"]) not in known]

        checked = _verify(proposed, snapshot, turns)

        if checked is None:
            # The check did not run: leave this snapshot (and later ones)
            # pending for the next update.
            _log.warning("State check unavailable for snapshot %s; will retry", snapshot["seq"])
            break

        verified, refused = checked
        rejected += refused

        replaced, answered = _consolidate(items, verified)

        for old_index, new_index in replaced:
            superseded.append({
                "text": items[old_index]["text"],
                "turn": items[old_index]["turn"],
                "by": verified[new_index]["text"],
            })

        drop = {old for old, _ in replaced} | answered
        items = [item for i, item in enumerate(items) if i not in drop] + verified
        items.sort(key=lambda item: item["turn"])

        covered.append(snapshot["state_hash"])

    if len(covered) == (len(versions[-1]["covered"]) if versions else 0):
        return None

    version = {
        "version": len(versions) + 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "covered": covered,
        "items": items,
        "superseded": superseded,
        "rejected": rejected,
    }

    append_state_version(user_id, chat_id, version)

    _log.info(
        "State v%d of chat %s: %d items, %d superseded, %d rejected",
        version["version"], chat_id, len(items), len(superseded), len(rejected),
    )

    return version


# ==========================================================
# Prompt blocks
# ==========================================================

def state_block(user_id: str, chat_id: str, max_tokens: int | None = None) -> str:
    """The state for the prompt ("" when there is none), newest items first
    under the token budget, shown oldest first."""

    items = current_state(user_id, chat_id)

    if not items:
        return ""

    budget = (max_tokens if max_tokens is not None else settings.STATE_MAX_TOKENS) * CHARS_PER_TOKEN
    lines, used = [], 0

    for item in sorted(items, key=lambda item: item["turn"], reverse=True):

        line = "- " + render_item(item)

        if lines and used + len(line) > budget:
            break

        lines.append(line)
        used += len(line)

    return f"\n\n{STATE_HEADER}\n" + "\n".join(reversed(lines))


def build_handoff(user_id: str, chat_id: str) -> str:
    """Where the conversation stands, in at most HANDOFF_MAX_TOKENS."""

    turns = get_turns(user_id, chat_id)
    snapshots = list_snapshots(user_id, chat_id)
    items = current_state(user_id, chat_id)

    if not turns:
        return ""

    documents = sorted({t["document"] for t in turns if t.get("document")})

    lines = [
        f"HAND-OFF: {len(turns)} turns so far, {len(snapshots)} snapshots"
        + (f", documents: {', '.join(documents)}" if documents else ""),
        f"Last question (turn {turns[-1]['turn']}): {' '.join(turns[-1]['question'].split())[:200]}",
    ]

    if items:
        lines.append("What the user stated or decided:")
        lines += [f"- {render_item(item)}" for item in items]

    asked = [a for s in snapshots for a in s["facets"].get("asked", [])]
    covered = snapshots[-1]["to_turn"] if snapshots else 0
    asked += [{"turn": t["turn"], "text": " ".join(t["question"].split())[:160]} for t in turns if t["turn"] > covered]

    if asked:
        lines.append("Recent topics asked:")
        lines += [f"- turn {a['turn']}: {a['text']}" for a in asked[-8:]]

    names = Counter(e["name"] for s in snapshots for e in s["facets"].get("entities", []))

    if names:
        lines.append("Key entities: " + "; ".join(name for name, _ in names.most_common(12)))

    return "\n".join(lines)[: settings.HANDOFF_MAX_TOKENS * CHARS_PER_TOKEN]


def is_resumed(user_id: str, chat_id: str, now: float | None = None) -> bool:
    """True when the chat has turns and the last one is older than
    SESSION_GAP_SECONDS: the user is coming back to it."""

    turns = get_turns(user_id, chat_id)

    if not turns or not turns[-1].get("ts"):
        return False

    return (now if now is not None else time.time()) - turns[-1]["ts"] > settings.SESSION_GAP_SECONDS


def handoff_block(user_id: str, chat_id: str) -> str:
    """The hand-off for the first turn of a resumed session, else ""."""

    if not is_resumed(user_id, chat_id):
        return ""

    text = build_handoff(user_id, chat_id)

    return f"\n\nHAND-OFF FROM THE EARLIER SESSION (use it to recall where the user left off):\n{text}" if text else ""
