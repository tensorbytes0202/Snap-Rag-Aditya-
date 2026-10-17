"""Query planning: one LLM call turns a question into a query type and a few
short, focused search queries.

A long multi-part question (a scenario plus six sub-questions plus rules about
citations) makes a poor search query: its embedding, its BM25 terms and the
cross-encoder's 512-token window are all spent on the story and the
instructions instead of the topics. Planning splits it into one short query per
topic, and retrieval then runs once per query.
"""

import logging
import re

from app.services.llm_provider import chat
from app.utils.json_utils import extract_json


_log = logging.getLogger(__name__)

MAX_SUBQUERIES = 6
LONG_QUESTION_WORDS = 40
MAX_QUESTION_CHARS = 6000

VALID_QUERY_TYPES = {
    "DEFINITION", "EXPLANATION", "COMPARISON", "LIST",
    "PROCEDURE", "SUMMARY", "CODE", "FACTUAL", "CONVERSATION",
}

SYSTEM_PROMPT = """
You plan retrieval for a document question-answering system.

Return ONLY valid JSON:

{
    "query_type": "DEFINITION | EXPLANATION | COMPARISON | LIST | PROCEDURE | SUMMARY | CODE | FACTUAL | CONVERSATION",
    "subqueries": ["...", "..."]
}

Rules:

- subqueries: 1 to 6 short search queries (under 15 words each), one for every
  distinct topic the question asks about.
- Write them in the vocabulary the document itself would use (doctrines,
  technical terms, case names, section titles), not the hypothetical story. Use
  the standard textbook name of a topic ("concurring cause", "last clear
  chance"), not a paraphrase ("multiple contributing causes").
- Do not add the author's name, the book title or filler such as "section"
  or "law of torts"; every word of a query must be a topic term.
- Ignore instructions about answer format, citations, constraints and sources.
- When the question has labelled or numbered parts (Q1, Q2, "Causation:",
  "Natural intervention:"), give every part its own subquery, even when two
  parts are related.
- A simple question gets exactly one subquery.
- CONVERSATION is for a question about the chat itself: what the user said, asked, decided or
  was told earlier ("what is my deadline?", "which sections did I skip?", "what did you say
  about X earlier?"), and also for a statement the user makes about themselves or their task
  that is not a question about the document ("My deadline is 15 November. Note it."). Its
  subquery names the topic being recalled or stated.
"""

# Sentences that only instruct the answerer; they say nothing about the topic.
_INSTRUCTION = re.compile(
    r"uploaded pdf|do not invent|use only|strict|constraint|distinguish|"
    r"send me|evaluate|clearly state|if the (source|pdf|retrieved|available)|"
    r"insufficient information|answer must",
    re.IGNORECASE,
)

_ASKING = re.compile(
    r"\?|^\s*(explain|analy[sz]e|determine|identify|describe|provide|how|what|"
    r"does|do|under|which|who|when|why)\b",
    re.IGNORECASE,
)

_SENTENCE = re.compile(r"(?<=[.?!:])\s+")


# A question about the conversation itself, not the document.
_CONVERSATIONAL = re.compile(
    r"\b(i|we)\s+(said|told|asked|mentioned|decided|chose|picked|gave|stated|wanted)\b|"
    r"\b(my|our)\s+(deadline|report|draft|decision|choice|preference|example|word limit)\b|"
    r"\byou\s+(said|told|mentioned|gave|cited|listed|explained)\b|"
    r"\b(so far|earlier|previously|before in this|at turn \d+|last (question|answer))\b|"
    r"\b(note|remember|keep in mind)\s+(this|that|it)\b|\b(note|remember) for later\b|"
    r"\b(my|our)\s+(\w+\s+){0,2}(is|are|will be|was|moved|changed)\b",
    re.IGNORECASE,
)


def guess_query_type(question: str) -> str:
    """No-LLM guess, used when planning fails."""

    return "CONVERSATION" if _CONVERSATIONAL.search(question) else "EXPLANATION"


def fallback_subqueries(question: str) -> list[str]:
    """No-LLM plan: the question itself when it is short; for a long one the
    sentences that ask something, minus the answer-format instructions."""

    if len(question.split()) <= LONG_QUESTION_WORDS:
        return [question]

    queries = []

    for sentence in _SENTENCE.split(question):

        sentence = " ".join(sentence.split())

        if not sentence or _INSTRUCTION.search(sentence) or not _ASKING.search(sentence):
            continue

        queries.append(" ".join(sentence.split()[:30]))

    queries = list(dict.fromkeys(queries))[:MAX_SUBQUERIES]

    return queries or [" ".join(question.split()[:40])]


def plan_query(question: str) -> dict:
    """{"query_type": ..., "subqueries": [...]}. Never raises: any LLM or
    parsing failure falls back to the heuristic plan."""

    plan = {"query_type": guess_query_type(question), "subqueries": fallback_subqueries(question)}

    try:

        response = chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question[:MAX_QUESTION_CHARS]},
            ],
            temperature=0,
        )

        result = extract_json(response["message"]["content"].strip())

    except Exception as exc:

        _log.warning("Query planning failed (%s); using the heuristic plan", exc)

        return plan

    if not isinstance(result, dict):

        _log.warning("Query planning returned no JSON; using the heuristic plan")

        return plan

    query_type = str(result.get("query_type", "")).upper().strip()

    if query_type in VALID_QUERY_TYPES:
        plan["query_type"] = query_type

    subqueries = [
        " ".join(q.split())
        for q in result.get("subqueries") or []
        if isinstance(q, str) and q.strip()
    ]

    if subqueries:
        plan["subqueries"] = list(dict.fromkeys(subqueries))[:MAX_SUBQUERIES]

    return plan
