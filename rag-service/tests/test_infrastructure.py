import threading

import pytest

from app.core import metrics
from app.core.config import settings
from app.services import rate_limiter, session_store
from app.utils.parallel import run_parallel


# ---------- metrics ----------

def test_counters_and_prometheus_text():
    metrics.inc("snaprag_queries_total", outcome="answered")
    metrics.inc("snaprag_queries_total", outcome="answered")
    metrics.observe("snaprag_query_seconds", 1.5)

    text = metrics.render_prometheus()

    assert 'snaprag_queries_total{outcome="answered"} 2' in text
    assert "snaprag_query_seconds_sum 1.500000" in text
    assert "snaprag_query_seconds_count 1" in text


def test_llm_calls_are_counted_inside_the_scope_including_worker_threads():
    with metrics.count_llm_calls() as calls:
        metrics.record_llm_call()
        run_parallel(metrics.record_llm_call, metrics.record_llm_call)

    metrics.record_llm_call()   # outside the scope: not counted in `calls`

    assert calls["n"] == 3
    assert metrics.value("snaprag_llm_calls_total", kind="chat") == 4


# ---------- run_parallel ----------

def test_run_parallel_returns_results_in_order():
    assert run_parallel(lambda: 1, lambda: 2, lambda: 3) == [1, 2, 3]
    assert run_parallel(lambda: "only") == ["only"]


def test_run_parallel_runs_tasks_concurrently():
    barrier = threading.Barrier(2, timeout=5)

    # Both tasks must be running at the same time to pass the barrier.
    assert run_parallel(lambda: barrier.wait() >= 0, lambda: barrier.wait() >= 0) == [True, True]


def test_run_parallel_raises_the_first_error_after_all_finish():
    done = []

    def slow_ok():
        done.append(1)
        return 1

    def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        run_parallel(boom, slow_ok)

    assert done == [1]


# ---------- session store ----------

def test_turns_are_numbered_and_append_only(chat_factory):
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2")])

    turns = session_store.get_turns(user, chat)

    assert [t["turn"] for t in turns] == [1, 2]
    assert turns[1]["question"] == "q2"


def test_malformed_chat_ids_never_reach_the_filesystem():
    assert session_store.get_turns("u1", "../../etc/passwd") == []
    assert not session_store.chat_exists("u1", "../x")


def test_chats_are_private_to_their_user(chat_factory):
    _, chat = chat_factory([("q", "a")], user_id="alice")

    assert not session_store.chat_exists("bob", chat)
    assert session_store.get_turns("bob", chat) == []


def test_partial_line_after_a_crash_is_skipped_and_the_next_turn_survives(chat_factory):
    user, chat = chat_factory([("q1", "a1")])

    path = session_store._chat_path(user, chat)
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"turn": 2, "question": "cut off')       # crash mid-write

    session_store.append_turn(user, chat, "q3", "a3")

    assert [t["question"] for t in session_store.get_turns(user, chat)] == ["q1", "q3"]


# ---------- request rate ----------

def test_request_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "QUERIES_PER_MINUTE", 3)
    rate_limiter._recent.clear()

    assert [rate_limiter.allow_request("u", now=t) for t in (0, 1, 2)] == [True] * 3
    assert rate_limiter.allow_request("u", now=3) is False
    assert rate_limiter.allow_request("other", now=3) is True      # per user
    assert rate_limiter.allow_request("u", now=61) is True         # the window moved on


def test_request_rate_limit_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "QUERIES_PER_MINUTE", 0)

    assert all(rate_limiter.allow_request("u", now=0) for _ in range(100))
