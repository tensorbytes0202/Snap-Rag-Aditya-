"""Streaming query as a LangGraph:

        retrieve ─┐
                  ├─► generate ─► check
        recall  ──┘

Document retrieval and memory recall only share the question, so they run
side by side; generate waits for both, fuses the memory into the context and
streams the tokens. The answer cannot be checked before it is written, so
check runs after the stream: it verifies the answer and scores it, and when the
confidence is below the threshold it reads more batches of evidence (AB-RAG, see
abrag) and sends the better answer as a `revised` event for the client to show instead.

Retrieval is query_runner.run_retrieval, memory is memory_retriever.recall_memory
(chats only) and the tokens come from generation_service.stream_answer. Nodes
emit plain event dicts (step / citations / token / verification / revised /
error); the route adds the SSE framing.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from app.core import metrics
from app.core.config import settings
from app.services.abrag import run_batches
from app.services.answering import check_answer, full_context, make_attempt, present_answer
from app.services.generation_service import stream_answer
from app.services.memory_retriever import finalize_memory, recall_memory
from app.services.query_runner import run_retrieval
from app.services.retrieval_pipeline import build_citations


_log = logging.getLogger(__name__)


class State(TypedDict, total=False):

    question: str
    active_document: str | None
    history: list
    session: tuple | None
    found: object
    memory: dict | None
    answer: str


def retrieve(state: State, writer: StreamWriter):

    def on_step(message, **detail):
        writer({"type": "step", "message": message, **({"detail": detail} if detail else {})})

    found = run_retrieval(state["question"], state.get("active_document"), on_step)

    if not found.results:
        writer({"type": "error", "message": "No relevant information found in the selected document."})
        return {}

    if not found.ranked_chunks:
        writer({"type": "error", "message": "No relevant information found."})
        return {}

    # Page anchors for the chat record (snapshots cite them later).
    writer({"type": "citations", "citations": build_citations(found.sources)})

    return {"found": found}


def recall(state: State, writer: StreamWriter):
    """Verified snapshot memory and conversation state (chats only)."""

    if not state.get("session"):
        return {}

    writer({"type": "step", "message": "Recalling verified memory of earlier turns"})

    return {"memory": recall_memory(*state["session"], state["question"], state.get("active_document"))}


def generate(state: State, writer: StreamWriter):

    found = state.get("found")

    if found is None:
        return {}

    memory = finalize_memory(state.get("memory"), found.top_context)

    if memory is not None:
        writer({"type": "step", "message": "Memory recalled", "detail": {
            "lambda": memory["lambda"],
            "snapshots": len(memory["snapshots"]),
            "sentences": len(memory["sentences"]),
            "state_items": memory.get("state_items", 0),
        }})

    context = full_context(found, memory)

    writer({"type": "step", "message": "Generating answer with LLM"})

    tokens = []

    try:
        for token in stream_answer(state["question"], context, found.query_type, state.get("history")):
            tokens.append(token)
            writer({"type": "token", "token": token})

    except Exception as exc:
        _log.warning("Stream generation error: %s", exc)
        writer({"type": "error", "message": str(exc)})
        return {}

    return {"memory": memory, "answer": "".join(tokens)}


def check(state: State, writer: StreamWriter):
    """Verify and score the streamed answer; read more evidence if it is not confident enough."""

    answer, found = state.get("answer"), state.get("found")

    if not answer or found is None:
        return {}

    question = state["question"]

    writer({"type": "step", "message": "Verifying the answer"})

    try:
        first = check_answer(question, answer, found, state.get("memory"), "initial")
    except Exception as exc:
        # The answer was already delivered; an unavailable check only means
        # it stays unverified.
        _log.warning("Stream verification failed: %s", exc)
        return {}

    writer({
        "type": "verification",
        "supported": first.supported,
        "confidence": first.confidence,
        "signals": first.signals,
        "verification": first.verification,
    })

    if first.confidence >= settings.STOP_CONFIDENCE:
        return {}

    writer({"type": "step", "message": "Confidence is low; reading more evidence", "detail": {"confidence": first.confidence}})

    try:
        best, attempts = run_batches(
            question,
            first,
            lambda **overrides: run_retrieval(question, state.get("active_document"), **overrides),
            lambda deeper, strategy: make_attempt(question, deeper, state.get("memory"), state.get("history"), strategy),
        )
    except Exception as exc:
        _log.warning("Stream extra batches failed: %s", exc)
        return {}

    if best is not first:
        metrics.inc("snaprag_revised_answers_total")
        writer({
            "type": "revised",
            "answer": present_answer(best),
            "supported": best.supported,
            "confidence": best.confidence,
            "signals": best.signals,
            "citations": build_citations(best.found.sources),
            "attempts": attempts,
        })

    return {}


_graph = (
    StateGraph(State)
    .add_node("retrieve", retrieve)
    .add_node("recall", recall)
    .add_node("generate", generate)
    .add_node("check", check)
    .add_edge(START, "retrieve")
    .add_edge(START, "recall")
    .add_edge(["retrieve", "recall"], "generate")
    .add_edge("generate", "check")
    .add_edge("check", END)
    .compile(name="rag_query_stream")
)


def stream_query(question: str, active_document: str | None = None, history=None, session=None):
    """`session` = (user_id, chat_id) of a chat, for memory recall."""

    yield from _graph.stream(
        {
            "question": question,
            "active_document": active_document,
            "history": history or [],
            "session": session,
        },
        stream_mode="custom",
    )
