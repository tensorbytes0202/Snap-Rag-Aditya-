"""Generate, check and present an answer: the pieces the batch and the streaming
query share around the AB-RAG loop (see abrag).
"""

from app.core.config import settings
from app.services.abrag import Attempt, score_answer
from app.services.followup_service import generate_followups
from app.services.generation_service import generate_answer
from app.services.memory_retriever import memory_block
from app.services.verification_service import verify_answer
from app.utils.parallel import run_parallel


UNVERIFIED_ANSWER = (
    "⚠️ The generated answer "
    "could not be fully verified "
    "from the retrieved document."
)


def full_context(found, memory) -> str:
    """Document passages plus the memory block."""

    return found.top_context + memory_block(memory)


def memory_texts(memory) -> list[str]:
    """The verified memory an answer may be drawn from (evidence for S2)."""

    if not memory:
        return []

    return [item["text"] for item in memory["sentences"]] + ([memory["state_text"]] if memory.get("state_text") else [])


def check_answer(question: str, answer: str, found, memory, strategy: str) -> Attempt:
    """Verify `answer` against the fused context and score it."""

    verification = verify_answer(question, answer, full_context(found, memory))
    confidence, signals = score_answer(found, answer, verification, memory_texts(memory))

    return Attempt(strategy, answer, verification, confidence, found, signals)


def make_attempt(question: str, found, memory, history, strategy: str, with_followups: bool = False) -> Attempt:
    """One full attempt: generate from the fused context, then check it. The
    follow-up questions are written beside the check, only when asked for
    (a deeper attempt that does not win would waste them)."""

    answer = generate_answer(question, full_context(found, memory), found.query_type, history)

    if not with_followups:
        return check_answer(question, answer, found, memory, strategy)

    attempt, followups = run_parallel(
        lambda: check_answer(question, answer, found, memory, strategy),
        lambda: generate_followups(question, answer),
    )

    attempt.followups = followups

    return attempt


def flag_unverified(answer: str, verification: dict) -> str:
    """Keep the answer, but say which statements the verifier could not
    confirm from the retrieved text."""

    claims = verification.get("unsupported_claims")

    listed = "\n".join(f"- {claim}" for claim in claims if isinstance(claim, str)) if isinstance(claims, list) else ""

    return (
        f"{answer}\n\n---\n"
        "⚠️ Parts of this answer could not be confirmed from the retrieved text"
        + (f":\n{listed}" if listed else ".")
    )


def present_answer(attempt: Attempt) -> str:
    """The answer to show: as generated when supported, otherwise flagged (or
    replaced, with DROP_UNVERIFIED_ANSWERS)."""

    if attempt.supported:
        return attempt.answer

    return UNVERIFIED_ANSWER if settings.DROP_UNVERIFIED_ANSWERS else flag_unverified(attempt.answer, attempt.verification)
