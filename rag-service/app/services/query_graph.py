"""Streaming query as a LangGraph: retrieve -> recall -> generate.

Only streaming lives here. Retrieval is query_runner.run_retrieval, memory
is memory_retriever.recall_memory (chats only) and the tokens come from
generation_service.stream_answer. Nodes emit plain event dicts
(step / citations / token / error); the route adds the SSE framing.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from app.services.generation_service import stream_answer
from app.services.memory_retriever import memory_block, recall_memory
from app.services.query_runner import run_retrieval
from app.services.retrieval_pipeline import build_citations


_log = logging.getLogger(__name__)


class State(TypedDict, total=False):

    question: str
    active_document: str | None
    history: list
    session: tuple | None
    query_type: str
    context: str


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

    return {"query_type": found.query_type, "context": found.top_context}


def recall(state: State, writer: StreamWriter):
    """Fuse verified snapshot memory into the context (chats only)."""

    if not state.get("context") or not state.get("session"):
        return {}

    writer({"type": "step", "message": "Recalling verified memory of earlier turns"})

    memory = recall_memory(*state["session"], state["question"], state.get("active_document"), state["context"])

    if memory:
        writer({"type": "step", "message": "Memory recalled", "detail": {
            "lambda": memory["lambda"],
            "snapshots": len(memory["snapshots"]),
            "sentences": len(memory["sentences"]),
        }})

    return {"context": state["context"] + memory_block(memory)}


def generate(state: State, writer: StreamWriter):

    if not state.get("context"):
        return {}

    writer({"type": "step", "message": "Generating answer with LLM"})

    try:
        for token in stream_answer(
            state["question"], state["context"], state["query_type"], state.get("history")
        ):
            writer({"type": "token", "token": token})

    except Exception as exc:
        _log.warning("Stream generation error: %s", exc)
        writer({"type": "error", "message": str(exc)})

    return {}


_graph = (
    StateGraph(State)
    .add_node("retrieve", retrieve)
    .add_node("recall", recall)
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "recall")
    .add_edge("recall", "generate")
    .add_edge("generate", END)
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
