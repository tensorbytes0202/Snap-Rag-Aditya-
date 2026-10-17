"""Context Quality Monitor (CQM): decides when to take a snapshot.

Measured over the window, i.e. the chat turns not yet in a snapshot:

    grounding      share of verified answers (turns without a verification
                   result, e.g. streamed ones, are left out; none -> 1.0)
    headroom       1 - window tokens / MEMORY_BUDGET_TOKENS (the "token" term;
                   high = plenty of room, so it adds to quality)
    coherence      mean cosine similarity of consecutive turns (one turn -> 1.0)
    contradiction  1.0 if the latest answer conflicts with an earlier one

    CIS = w_g*grounding + w_t*headroom + w_c*coherence - w_x*contradiction

Trigger: headroom < CQM_TRIGGER_HEADROOM ("pressure") or CIS < CQM_TRIGGER_CIS
("cis"), once the window holds at least MIN_WINDOW_TURNS turns.
"""

import logging

from langsmith import traceable

from app.api.dependencies.embedding import get_embedding_model
from app.core.config import settings
from app.services.llm_provider import chat
from app.utils.json_utils import extract_json


_log = logging.getLogger(__name__)

# A snapshot of a single turn compresses nothing.
MIN_WINDOW_TURNS = 2

# Characters per token, the usual rough estimate for English text.
CHARS_PER_TOKEN = 4

_CONTRADICTION_PROMPT = """Below are EARLIER answers from one conversation and the LATEST answer.
Does the LATEST answer state something that conflicts with an EARLIER answer, i.e. a
different value, name or claim about the same thing? Different topics are not conflicts.

EARLIER:
{earlier}

LATEST (turn {turn}):
{latest}

Return ONLY JSON: {{"conflicts": [{{"earlier_turn": <number>, "what": "<one line>"}}]}}
Use an empty list when there is no conflict."""

# Long answers are cut to this many characters for the contradiction check.
_ANSWER_CHARS = 600


def turn_tokens(turn: dict) -> int:

    return (len(turn["question"]) + len(turn["answer"])) // CHARS_PER_TOKEN


def _grounding(window: list[dict]) -> float:

    known = [turn["verified"] for turn in window if isinstance(turn.get("verified"), bool)]

    return sum(known) / len(known) if known else 1.0


def _coherence(window: list[dict]) -> float:

    if len(window) < 2:
        return 1.0

    vectors = get_embedding_model().encode(
        [f"{turn['question']}\n{turn['answer'][:500]}" for turn in window],
        normalize_embeddings=True,
    )

    similarities = [float(a @ b) for a, b in zip(vectors, vectors[1:])]

    return min(1.0, max(0.0, sum(similarities) / len(similarities)))


def _contradiction(window: list[dict]) -> tuple[float, list | None]:
    """(1.0 or 0.0, the conflicts found). A failed check counts as no
    conflict and returns None, so it never triggers a snapshot by itself."""

    if len(window) < 2 or not settings.CQM_CHECK_CONTRADICTION:
        return 0.0, []

    *earlier, latest = window

    prompt = _CONTRADICTION_PROMPT.format(
        earlier="\n".join(f"[turn {t['turn']}] {t['answer'][:_ANSWER_CHARS]}" for t in earlier),
        turn=latest["turn"],
        latest=latest["answer"][:_ANSWER_CHARS],
    )

    try:
        data = extract_json(chat([{"role": "user", "content": prompt}])["message"]["content"])
        conflicts = data.get("conflicts") if isinstance(data, dict) else None
    except Exception:
        _log.warning("Contradiction check failed; counting it as no conflict", exc_info=True)
        return 0.0, None

    if not isinstance(conflicts, list):
        return 0.0, None

    return (1.0 if conflicts else 0.0), conflicts


@traceable(name="cqm_measure", run_type="chain")
def measure(window: list[dict]) -> dict:

    tokens = sum(turn_tokens(turn) for turn in window)

    grounding = _grounding(window)
    headroom = min(1.0, max(0.0, 1 - tokens / settings.MEMORY_BUDGET_TOKENS))
    coherence = _coherence(window)
    contradiction, conflicts = _contradiction(window)

    cis = (
        settings.CQM_W_GROUNDING * grounding
        + settings.CQM_W_TOKEN * headroom
        + settings.CQM_W_COHERENCE * coherence
        - settings.CQM_W_CONTRADICTION * contradiction
    )

    return {
        "tokens": tokens,
        "grounding": round(grounding, 3),
        "headroom": round(headroom, 3),
        "coherence": round(coherence, 3),
        "contradiction": contradiction,
        "conflicts": conflicts,
        "cis": round(cis, 3),
    }


def decide(signals: dict, window_turns: int) -> str | None:
    """"pressure", "cis" or None (no snapshot)."""

    if window_turns < MIN_WINDOW_TURNS:
        return None

    # Ablation baseline: a fixed schedule instead of the quality signals.
    if settings.CQM_FIXED_INTERVAL > 0:
        return "interval" if window_turns >= settings.CQM_FIXED_INTERVAL else None

    if signals["headroom"] < settings.CQM_TRIGGER_HEADROOM:
        return "pressure"

    if signals["cis"] < settings.CQM_TRIGGER_CIS:
        return "cis"

    return None
