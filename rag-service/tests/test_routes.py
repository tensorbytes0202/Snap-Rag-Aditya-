"""HTTP layer and the streaming graph, with the pipeline stubbed."""

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.routes import query as query_routes
from app.core.config import settings
from app.core.security import get_current_user_id
from app.services import query_graph, session_store, snapshot_store
from app.services.abrag import Attempt

import main


@pytest.fixture
def client(monkeypatch):

    main.app.dependency_overrides[get_current_user_id] = lambda: "alice"
    monkeypatch.setattr(query_routes, "get_active_document", lambda user_id: "d.pdf")
    monkeypatch.setattr(query_routes, "after_turn", lambda user_id, chat_id: None)

    yield TestClient(main.app)

    main.app.dependency_overrides.clear()


def events_of(response):
    return [json.loads(block[6:]) for block in response.text.split("\n\n") if block.startswith("data: ")]


# ---------- chats ----------

def test_chat_lifecycle_and_privacy(client):
    chat_id = client.post("/chats").json()["chat_id"]

    assert client.get(f"/chats/{chat_id}/messages").json()["turns"] == []
    assert any(c["chat_id"] == chat_id for c in client.get("/chats").json()["chats"])

    main.app.dependency_overrides[get_current_user_id] = lambda: "mallory"

    assert client.get(f"/chats/{chat_id}/messages").status_code == 404
    assert client.get(f"/chats/{chat_id}/memory").status_code == 404
    assert client.get(f"/chats/{chat_id}/handoff").status_code == 404


def test_memory_endpoint_shows_state_and_handoff_endpoint_the_document(client):
    chat_id = client.post("/chats").json()["chat_id"]
    session_store.append_turn("alice", chat_id, "My deadline is 15 November", "Noted.")
    snapshot_store.append_state_version("alice", chat_id, {"version": 1, "covered": [], "items": [
        {"kind": "user_fact", "text": "The deadline is 15 November", "turn": 1}]})

    memory = client.get(f"/chats/{chat_id}/memory").json()
    handoff = client.get(f"/chats/{chat_id}/handoff").json()

    assert memory["state"][0]["text"] == "The deadline is 15 November" and memory["state_versions"] == 1
    assert "The deadline is 15 November" in handoff["handoff"]


# ---------- /query ----------

def answer_response(**extra):
    return {"answer": "The answer.", "citations": [{"source_id": 1, "document": "d.pdf", "page": 3}],
            "verification": {"supported": True}, "confidence": 0.9, **extra}


def test_query_records_the_turn_with_verification_and_confidence(client, monkeypatch):
    seen = {}

    def process(question, document, history, session):
        seen.update(history=history, session=session)
        return answer_response()

    monkeypatch.setattr(query_routes, "process_query", process)
    chat_id = client.post("/chats").json()["chat_id"]

    assert client.post("/query", json={"question": "What is X?", "chat_id": chat_id}).status_code == 200

    turn = session_store.get_turns("alice", chat_id)[0]

    assert turn["verified"] is True and turn["confidence"] == 0.9 and turn["document"] == "d.pdf"
    assert turn["citations"][0]["page"] == 3 and seen["session"] == ("alice", chat_id)


def test_query_without_a_chat_stores_nothing(client, monkeypatch):
    monkeypatch.setattr(query_routes, "process_query", lambda *a: answer_response())

    assert client.post("/query", json={"question": "q"}).status_code == 200
    assert session_store.list_chats("alice") == []


def test_query_on_a_missing_chat_is_404(client, monkeypatch):
    monkeypatch.setattr(query_routes, "process_query", lambda *a: pytest.fail("must not run"))

    assert client.post("/query", json={"question": "q", "chat_id": "0" * 32}).status_code == 404


def test_the_prompt_history_includes_turns_after_the_last_snapshot(client, monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_TURNS", 1)
    monkeypatch.setattr(settings, "HISTORY_MAX_TOKENS", 10_000)
    seen = {}
    monkeypatch.setattr(query_routes, "process_query", lambda q, d, history, s: seen.update(h=history) or answer_response())
    chat_id = client.post("/chats").json()["chat_id"]
    for i in range(1, 5):
        session_store.append_turn("alice", chat_id, f"q{i}", f"a{i}")

    client.post("/query", json={"question": "next", "chat_id": chat_id})

    assert [t["turn"] for t in seen["h"]] == [1, 2, 3, 4]


def test_a_failed_answer_refunds_nothing_for_full_users_and_maps_llm_errors(client, monkeypatch):
    from app.core.exceptions import LLMProviderError

    def fail(*a):
        raise LLMProviderError("down")

    monkeypatch.setattr(query_routes, "process_query", fail)

    assert client.post("/query", json={"question": "q"}).status_code == 502


# ---------- /query/stream ----------

def fake_stream(events):
    return lambda question, document, history, session: iter(events)


def test_stream_records_verification_and_confidence(client, monkeypatch):
    monkeypatch.setattr(query_routes, "stream_query", fake_stream([
        {"type": "citations", "citations": [{"source_id": 1, "document": "d.pdf", "page": 2}]},
        {"type": "token", "token": "Hello "}, {"type": "token", "token": "world"},
        {"type": "verification", "supported": True, "confidence": 0.88},
    ]))
    chat_id = client.post("/chats").json()["chat_id"]

    response = client.post("/query/stream", json={"question": "q", "chat_id": chat_id})

    assert [e["type"] for e in events_of(response)] == ["citations", "token", "token", "verification", "done"]

    turn = session_store.get_turns("alice", chat_id)[0]

    assert turn["answer"] == "Hello world" and turn["verified"] is True and turn["confidence"] == 0.88


def test_a_revised_answer_replaces_the_streamed_one_in_the_record(client, monkeypatch):
    monkeypatch.setattr(query_routes, "stream_query", fake_stream([
        {"type": "token", "token": "weak answer"},
        {"type": "verification", "supported": False, "confidence": 0.4},
        {"type": "revised", "answer": "better answer", "supported": True, "confidence": 0.9,
         "citations": [{"source_id": 1, "document": "d.pdf", "page": 9}]},
    ]))
    chat_id = client.post("/chats").json()["chat_id"]

    client.post("/query/stream", json={"question": "q", "chat_id": chat_id})

    turn = session_store.get_turns("alice", chat_id)[0]

    assert turn["answer"] == "better answer" and turn["verified"] is True
    assert turn["citations"][0]["page"] == 9


def test_a_stream_that_errors_is_not_recorded(client, monkeypatch):
    monkeypatch.setattr(query_routes, "stream_query", fake_stream([
        {"type": "token", "token": "partial"}, {"type": "error", "message": "boom"}]))
    chat_id = client.post("/chats").json()["chat_id"]

    client.post("/query/stream", json={"question": "q", "chat_id": chat_id})

    assert session_store.get_turns("alice", chat_id) == []


def test_an_unexpected_stream_failure_is_reported_as_an_event(client, monkeypatch):
    def broken(*a):
        raise RuntimeError("kaput")
        yield

    monkeypatch.setattr(query_routes, "stream_query", broken)

    kinds = [e["type"] for e in events_of(client.post("/query/stream", json={"question": "q"}))]

    assert kinds == ["error", "done"]


# ---------- the streaming graph ----------

def stub_graph(monkeypatch, tokens=("The ", "answer."), first=0.95, revised=None, check_error=None):
    """Patches the graph's collaborators; returns what was called."""

    called = SimpleNamespace(retrievals=[], checks=0)
    retrieved = SimpleNamespace(
        results=[1], ranked_chunks=[{"page": 1}], top_chunks=[{"page": 1, "rerank_score": 2.0}], top_context="DOC",
        query_type="EXPLANATION", sources=[{"document": "d.pdf", "page": 1}], plan={}, rewritten_question=["q"],
    )

    def run_retrieval(question, active_document=None, on_step=None, **overrides):
        called.retrievals.append(overrides)
        if on_step:
            on_step("Understood the question")
        return retrieved

    def check_answer(question, answer, found, memory, strategy):
        called.checks += 1
        if check_error:
            raise check_error
        return Attempt(strategy, answer, {"supported": first >= 0.8, "confidence": first}, first, found,
                       {"model": first})

    monkeypatch.setattr(query_graph, "run_retrieval", run_retrieval)
    monkeypatch.setattr(query_graph, "recall_memory", lambda *a: {
        "turn": 3, "lambda": 0.4, "snapshots": [1], "sentences": [], "tokens": 0, "state_items": 2,
        "state_text": "\n\nSTATE", "candidates": [{"text": "remembered", "source": "snapshot"}]})
    monkeypatch.setattr(query_graph, "stream_answer", lambda *a: iter(tokens))
    monkeypatch.setattr(query_graph, "check_answer", check_answer)

    if revised is not None:
        monkeypatch.setattr(query_graph, "run_batches", lambda q, f, r, a: (revised, [{"strategy": "deeper"}]))

    return called, retrieved


def run_graph(session=("alice", "c1")):
    return list(query_graph.stream_query("What is X?", "d.pdf", [], session))


def test_the_graph_streams_then_verifies(monkeypatch):
    stub_graph(monkeypatch)

    events = run_graph()
    kinds = [e["type"] for e in events]

    assert "".join(e["token"] for e in events if e["type"] == "token") == "The answer."
    assert kinds.index("citations") < kinds.index("token") < kinds.index("verification")
    assert events[-1] == {"type": "verification", "supported": True, "confidence": 0.95, "signals": {"model": 0.95},
                          "verification": {"supported": True, "confidence": 0.95}}
    assert not any(e["type"] == "revised" for e in events)


def test_the_graph_reports_the_recalled_memory(monkeypatch):
    stub_graph(monkeypatch)

    recalled = next(e for e in run_graph() if e.get("message") == "Memory recalled")

    assert recalled["detail"] == {"lambda": 0.4, "snapshots": 1, "sentences": 1, "state_items": 2}


def test_without_a_chat_there_is_no_recall(monkeypatch):
    stub_graph(monkeypatch)

    messages = [e.get("message") for e in run_graph(session=None)]

    assert "Recalling verified memory of earlier turns" not in messages and "Memory recalled" not in messages


def test_a_low_confidence_answer_is_revised(monkeypatch):
    better = Attempt("corrective", "Better answer.", {"supported": True}, 0.9,
                     SimpleNamespace(sources=[{"document": "d.pdf", "page": 7}]))
    stub_graph(monkeypatch, first=0.4, revised=better)

    events = run_graph()
    revised = next(e for e in events if e["type"] == "revised")

    assert revised["answer"] == "Better answer." and revised["confidence"] == 0.9
    assert revised["citations"][0]["page"] == 7 and revised["attempts"] == [{"strategy": "deeper"}]


def test_a_revision_that_is_no_better_is_not_sent(monkeypatch):
    called, retrieved = stub_graph(monkeypatch, first=0.4)

    def keep_first(question, first, retrieve, attempt_from):
        return first, [first.summary()]

    monkeypatch.setattr(query_graph, "run_batches", keep_first)

    assert not any(e["type"] == "revised" for e in run_graph())


def test_an_unavailable_check_leaves_the_delivered_answer_alone(monkeypatch):
    stub_graph(monkeypatch, check_error=RuntimeError("gateway down"))

    events = run_graph()

    assert [e for e in events if e["type"] in ("verification", "revised", "error")] == []


def test_nothing_found_ends_with_an_error_and_no_tokens(monkeypatch):
    called, retrieved = stub_graph(monkeypatch)
    retrieved.results = []

    events = run_graph()

    assert [e["type"] for e in events if e["type"] in ("error", "token", "verification")] == ["error"]
