"""The recent turns shown to the LLM so it can resolve follow-ups.

Turns older than the newest snapshot are covered by snapshot memory and turns
within HISTORY_TURNS are always shown. The turns in between (after the newest
snapshot, but not among the last few) would be in neither, so they are added
newest first until HISTORY_MAX_TOKENS is used. Answers are cut to
HISTORY_ANSWER_CHARS: the prompt only needs enough of them to say what was
discussed, and the full text stays in the chat record.
"""

from app.core.config import settings
from app.services.context_monitor import CHARS_PER_TOKEN
from app.services.session_store import get_turns
from app.services.snapshot_store import list_snapshots


def _shorten(turn: dict) -> dict:

    answer = turn["answer"]
    limit = settings.HISTORY_ANSWER_CHARS

    if len(answer) > limit:
        answer = answer[:limit].rstrip() + " [...]"

    return {"turn": turn["turn"], "question": turn["question"], "answer": answer}


def prompt_history(user_id: str, chat_id: str) -> list[dict]:
    """The chat's turns for the prompt, oldest first."""

    turns = get_turns(user_id, chat_id)

    if not turns:
        return []

    covered = 0

    try:
        snapshots = list_snapshots(user_id, chat_id)
        covered = snapshots[-1]["to_turn"] if snapshots else 0
    except ValueError:
        pass

    keep = {turn["turn"] for turn in turns[-settings.HISTORY_TURNS:]} if settings.HISTORY_TURNS > 0 else set()

    budget = settings.HISTORY_MAX_TOKENS * CHARS_PER_TOKEN
    used = sum(
        len(turn["question"]) + min(len(turn["answer"]), settings.HISTORY_ANSWER_CHARS)
        for turn in turns if turn["turn"] in keep
    )

    for turn in reversed(turns):

        if turn["turn"] <= covered or turn["turn"] in keep:
            continue

        size = len(turn["question"]) + min(len(turn["answer"]), settings.HISTORY_ANSWER_CHARS)

        if used + size > budget:
            break

        keep.add(turn["turn"])
        used += size

    return [_shorten(turn) for turn in turns if turn["turn"] in keep]
