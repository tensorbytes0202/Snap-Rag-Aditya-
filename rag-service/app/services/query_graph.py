"""Streaming query as a two-node LangGraph: retrieve -> generate.

Only streaming lives here. Retrieval is query_runner.run_retrieval and the
tokens come from generation_service.stream_answer. Nodes emit plain event
dicts (step / token / error); the route adds the SSE framing.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from app.services.generation_service import stream_answer
from app.services.query_runner import run_retrieval


_log = logging.getLogger(__name__)


class State(TypedDict, total=False):

    question: str
    active_document: str | None
    history: list
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

    return {"query_type": found.query_type, "context": found.top_context}


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
    .add_node("generate", generate)
    .add_edge(START, "retrieve")
    .add_edge("retrieve", "generate")
    .add_edge("generate", END)
    .compile(name="rag_query_stream")
)


def stream_query(question: str, active_document: str | None = None, history=None):

    yield from _graph.stream(
        {"question": question, "active_document": active_document, "history": history or []},
        stream_mode="custom",
    )
