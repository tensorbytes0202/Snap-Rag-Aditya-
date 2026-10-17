"""Adaptive budgeted retrieval (AB-RAG) with a verified-memory evidence pool.

Instead of reading a fixed top-k for every question, the first batch of passages
is answered and the answer is scored. If the score is below the threshold and
budget remains, the next batch is added and the answer is made again.

    confidence = W_MODEL * S1 + W_SIMILARITY * S2 + W_SPREAD * S3

    S1  model certainty         the checker's confidence in the answer. The gateway gives no
                                token probabilities, so this is a judged certainty (a proxy)
    S2  answer-evidence match   cosine similarity of the answer to the closest evidence text
    S3  reranker-score spread   standard deviation of the reranker scores of the passages read
                                (a clear winner among them means the ranking separated the evidence)

An answer the checker does not support is capped below the threshold, so it is never
accepted as it is. These signals are proxies for reliability, not proof.

What this adds to AB-RAG:
    * intent-aware budget: the first batch and the number of extra batches follow the
      question's intent (query_router), not one k
    * the evidence pool holds the verified conversation memory as well as the passages, so a
      question answered from memory is not sent back to the documents; a question about the
      conversation itself never is
    * the next batch is free while the ranking has passages left (no new search); only when
      it is used up does a fresh, wider search run, aimed by the checker's complaints

Retrieval and the LLM are passed in, which keeps the loop testable and shared by the batch
and the streaming query.
"""

import logging
import statistics
from dataclasses import dataclass, field

from langsmith import traceable

from app.api.dependencies.embedding import get_embedding_model
from app.core import metrics
from app.core.config import settings
from app.services.llm_provider import chat
from app.utils.json_utils import extract_json


_log = logging.getLogger(__name__)

# Cosine similarity below SIM_FLOOR scores 0 and SIM_FLOOR + SIM_RANGE or more scores 1
# (related texts of the embedding model sit around 0.6-0.8).
SIM_FLOOR, SIM_RANGE = 0.4, 0.4

# A reranker-score standard deviation of SPREAD_FULL or more scores 1.
SPREAD_FULL = 3.0

_REFINE_PROMPT = """A question was answered from passages retrieved from a document, but a checker
found problems with the answer.

QUESTION: {question}
PROBLEMS: {problems}

Write 1 to 3 short search queries (under 15 words each), in the vocabulary the document itself
would use, that would find the evidence the answer was missing.
Return ONLY JSON: {{"queries": ["...", "..."]}}"""


@dataclass
class Attempt:

    strategy: str
    answer: str
    verification: dict
    confidence: float
    found: object
    signals: dict = field(default_factory=dict)
    followups: list | None = None

    @property
    def supported(self) -> bool:
        return self.verification.get("supported") is True

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "confidence": self.confidence,
            "signals": self.signals,
            "supported": self.supported,
            "passages": len(self.found.top_chunks),
            "pages": sorted({c.get("page") for c in self.found.top_chunks if c.get("page") is not None}),
        }


# ==========================================================
# Confidence
# ==========================================================

def _unit(value) -> float:

    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def similarity_signal(answer: str, evidence: list[str]) -> float:
    """S2: the answer's cosine similarity to its closest evidence text, scaled to 0-1."""

    if not answer.strip() or not evidence:
        return 0.0

    vectors = get_embedding_model().encode([answer[:2000]] + [text[:2000] for text in evidence], normalize_embeddings=True)
    closest = max(float(vectors[0] @ other) for other in vectors[1:])

    return _unit((closest - SIM_FLOOR) / SIM_RANGE)


def spread_signal(scores: list[float]) -> float:
    """S3: standard deviation of the reranker scores, scaled to 0-1. One passage
    carries no spread, so it scores neutral."""

    if len(scores) < 2:
        return 0.5

    return _unit(statistics.pstdev(scores) / SPREAD_FULL)


def score_answer(found, answer: str, verification: dict, memory_texts: list[str] = ()) -> tuple[float, dict]:
    """(confidence, {"model", "similarity", "spread"}). Conversation questions
    rest on the checker alone: the documents say nothing about them."""

    model = _unit(verification.get("confidence"))

    if found.query_type == "CONVERSATION":
        confidence, signals = model, {"model": model}
    else:
        passages = [chunk.get("parent_text") or chunk["text"] for chunk in found.top_chunks]
        signals = {
            "model": model,
            "similarity": similarity_signal(answer, passages + list(memory_texts)),
            "spread": spread_signal([chunk["rerank_score"] for chunk in found.top_chunks]),
        }
        confidence = (
            settings.CONF_W_MODEL * signals["model"]
            + settings.CONF_W_SIMILARITY * signals["similarity"]
            + settings.CONF_W_SPREAD * signals["spread"]
        ) / (settings.CONF_W_MODEL + settings.CONF_W_SIMILARITY + settings.CONF_W_SPREAD)

    if verification.get("supported") is not True:
        confidence = min(confidence, settings.STOP_CONFIDENCE - 0.01)

    return round(confidence, 2), {name: round(value, 2) for name, value in signals.items()}


# ==========================================================
# The loop
# ==========================================================

@traceable(name="refine_queries", run_type="chain")
def refine_queries(question: str, attempt: Attempt) -> list[str]:
    """Search queries aimed at what the checker could not find; the unsupported
    claims themselves when the LLM is unavailable."""

    claims = [c for c in attempt.verification.get("unsupported_claims") or [] if isinstance(c, str)]
    problems = "; ".join(claims) or str(attempt.verification.get("reason") or "the answer was not supported")

    try:
        data = extract_json(chat(
            [{"role": "user", "content": _REFINE_PROMPT.format(question=question[:1500], problems=problems[:1500])}]
        )["message"]["content"])
        queries = data.get("queries") if isinstance(data, dict) else None
    except Exception:
        _log.warning("Query refinement failed; using the unsupported claims", exc_info=True)
        queries = None

    if not isinstance(queries, list):
        queries = [" ".join(claim.split()[:15]) for claim in claims[:3]]

    return [" ".join(q.split()) for q in queries if isinstance(q, str) and q.strip()][:3]


def run_batches(question: str, first: Attempt, retrieve, attempt_from):
    """(best attempt, log of every attempt). `retrieve(**overrides)` runs a
    fresh search; `attempt_from(found, strategy)` answers and scores from a
    Retrieval. The budget is the intent's, capped by MAX_EXTRA_BATCHES."""

    best, current, log = first, first, [first.summary()]

    budget = min(settings.MAX_EXTRA_BATCHES, first.found.config.get("batches", 0))

    for _ in range(budget):

        if best.confidence >= settings.STOP_CONFIDENCE:
            break

        found = current.found.deeper(max(2, current.found.limit // 2))
        strategy = "deeper"

        if found is None:
            strategy = "fresh"
            found = retrieve(plan=current.found.plan, extra_queries=refine_queries(question, current), wide=True)

            if not found.ranked_chunks:
                break

        metrics.inc("snaprag_extra_batches_total", strategy=strategy)

        current = attempt_from(found, strategy)
        log.append(current.summary())

        if current.confidence > best.confidence:
            best = current

    return best, log
