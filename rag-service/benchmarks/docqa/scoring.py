"""Deterministic scoring for the document-QA architecture comparison (no LLM judge).

An answerable question lists facts; each fact has a pattern for the answer ("a")
and optionally a pattern for the source text ("e", used to check the retrieved
context and validated against the gold pages). An answer is correct when every
fact is found. An unanswerable question (absent information or a false premise)
is handled when the answer declines or states the correction.
"""

import re

# Phrases of a refusal ("the document does not say ...").
DECLINE = (
    r"not\s+(mentioned|stated|specified|provided|given|described|discussed|said|available|found|included|covered|addressed|revealed|indicated|recorded)"
    r"|no\s+(mention|information|evidence|record|indication|details?|reference)"
    r"|(does|do|did|is|was|are|were|has|have)(\s+not|n.t)\s+(mention|say|state|specify|provide|contain|appear|exist|include|attend|name|give|describe|find|show|reveal|have|cover|address|indicate|record|reach|extend)"
    r"|cannot\s+(be\s+)?(found|determined|answered|verified)|can.t\s+(be\s+)?(found|determined|answered)"
    r"|unable\s+to\s+(find|determine|answer)|insufficient|could\s+not\s+be\s+fully\s+verified"
    r"|no\s+such|never\s+(mention|state|say|quote|quoted|given|included|shown|reproduced|provided)|not\s+told|false\s+premise|incorrect\s+premise|there\s+is\s+no\b"
    r"|no\s+relevant\s+information"
)


def norm(text) -> str:
    """Flattened text: scanned pages carry odd line breaks and broken quote marks."""

    return re.sub(r"\s+", " ", (text or "").replace("�", "'"))


def _has(pattern: str, text: str) -> bool:

    return re.search(pattern, text, re.I) is not None


def score_answer(item: dict, answer: str) -> dict:
    """{"correct", "score" (fraction of facts), "declined"}."""

    answer = norm(answer)
    declined = _has(DECLINE, answer)

    if item.get("answerable", True) is False:

        handled = declined or bool(item.get("correction")) and _has(item["correction"], answer)

        return {"correct": handled, "score": 1.0 if handled else 0.0, "declined": declined}

    hits = [_has(fact["a"], answer) for fact in item["facts"]]
    score = sum(hits) / len(hits)

    return {"correct": all(hits), "score": score, "declined": declined and score == 0}


def score_retrieval(item: dict, context: str, pages) -> dict | None:
    """Share of facts present in the context the LLM read, and whether it read a
    gold page. None for unanswerable questions (there is nothing to retrieve)."""

    if item.get("answerable", True) is False:
        return None

    context = norm(context)
    hits = [_has(fact.get("e", fact["a"]), context) for fact in item["facts"]]
    gold, read = set(item["pages"]), set(pages)

    return {
        "context_recall": sum(hits) / len(hits),
        "page_hit": bool(gold & read),
        "page_recall": len(gold & read) / len(gold),
    }
