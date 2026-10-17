"""Scores one answer to one probe. Pure functions: no LLM judge, so a score
depends only on the answer text and is the same on every run.

    recall / supersede / reference   every expected pattern occurs in the answer
    absent                           the answer says the user has not told it
                                     (a refusal), instead of inventing a value
    trap                             the answer corrects or declines the false
                                     premise, and does not endorse it

Patterns are case-insensitive regular expressions that match across lines.
"""

import re

FLAGS = re.I | re.S


def _found(pattern: str, text: str) -> bool:

    return re.search(pattern, text, FLAGS) is not None


def score_probe(probe: dict, answer: str, refusal: str) -> dict:
    """{"correct": bool, "reason": str}"""

    kind = probe["probe_kind"]

    if not answer or not answer.strip():
        return {"correct": False, "reason": "empty answer"}

    if kind in ("recall", "supersede", "reference"):

        missing = [p for p in probe.get("expect_all", []) if not _found(p, answer)]

        if not missing:
            return {"correct": True, "reason": "found"}

        if kind == "supersede" and any(_found(p, answer) for p in probe.get("stale", [])):
            return {"correct": False, "reason": "stale value"}

        if _found(refusal, answer):
            return {"correct": False, "reason": "forgot"}

        return {"correct": False, "reason": "wrong or missing: " + ", ".join(missing)}

    if kind == "absent":

        if _found(refusal, answer):
            return {"correct": True, "reason": "declined"}

        return {"correct": False, "reason": "invented an answer"}

    if kind == "trap":

        if any(_found(p, answer) for p in probe.get("forbid", [])):
            return {"correct": False, "reason": "endorsed the false premise"}

        if _found(refusal, answer) or any(_found(p, answer) for p in probe.get("ok_cues", [])):
            return {"correct": True, "reason": "corrected or declined"}

        return {"correct": False, "reason": "no correction"}

    raise ValueError(f"unknown probe kind: {kind}")


def page_hit(gold_pages, citations) -> bool | None:
    """Did the answer cite a page inside the gold range? None when the turn
    has no gold range."""

    if not gold_pages:
        return None

    low, high = gold_pages

    return any(isinstance(c.get("page"), int) and low <= c["page"] <= high for c in citations or [])
