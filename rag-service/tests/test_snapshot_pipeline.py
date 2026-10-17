"""Context monitor, snapshot engine and the history shown to the LLM."""

import json

import pytest

from app.core.config import settings
from app.services import context_monitor, history, session_store, snapshot_engine, snapshot_store
from app.services.snapshot_engine import _clean_facets, create_snapshot, render_snapshot

from conftest import snapshot_json


def turn(n, question="q", answer="a", **extra):
    return {"turn": n, "question": question, "answer": answer, **extra}


# ---------- context monitor ----------

def signals(headroom=0.9, cis=0.9):
    return {"headroom": headroom, "cis": cis}


def test_no_snapshot_for_a_single_turn_window():
    assert context_monitor.decide(signals(headroom=0.0, cis=0.0), 1) is None


def test_pressure_and_cis_triggers():
    assert context_monitor.decide(signals(headroom=0.1), 3) == "pressure"
    assert context_monitor.decide(signals(cis=0.4), 3) == "cis"
    assert context_monitor.decide(signals(), 3) is None


def test_pressure_wins_over_cis():
    assert context_monitor.decide(signals(headroom=0.1, cis=0.1), 3) == "pressure"


def test_fixed_interval_ignores_the_quality_signals(monkeypatch):
    monkeypatch.setattr(settings, "CQM_FIXED_INTERVAL", 4)

    assert context_monitor.decide(signals(headroom=0.0, cis=0.0), 3) is None
    assert context_monitor.decide(signals(), 4) == "interval"


def test_measure_combines_the_four_signals(llm, fake_embed):
    llm.reply("EARLIER answers", json.dumps({"conflicts": [{"earlier_turn": 1, "what": "x"}]}))
    window = [turn(1, "alpha beta", "alpha beta", verified=True), turn(2, "alpha beta", "alpha beta", verified=False)]

    result = context_monitor.measure(window)

    assert result["grounding"] == 0.5
    assert result["contradiction"] == 1.0
    assert result["coherence"] == pytest.approx(1.0)
    expected = 0.35 * 0.5 + 0.25 * result["headroom"] + 0.25 * 1.0 - 0.15 * 1.0
    assert result["cis"] == pytest.approx(expected, abs=1e-3)


def test_unreachable_contradiction_check_counts_as_no_conflict(llm, fake_embed):
    # no scripted reply -> the LLM call fails
    result = context_monitor.measure([turn(1), turn(2)])

    assert result["contradiction"] == 0.0 and result["conflicts"] is None


def test_contradiction_check_can_be_switched_off(llm, fake_embed, monkeypatch):
    monkeypatch.setattr(settings, "CQM_CHECK_CONTRADICTION", False)

    context_monitor.measure([turn(1), turn(2)])

    assert llm.calls == []


def test_turns_without_a_verification_result_do_not_count_against_grounding():
    assert context_monitor._grounding([turn(1, verified=None), turn(2, verified=None)]) == 1.0
    assert context_monitor._grounding([turn(1, verified=True), turn(2, verified=None)]) == 1.0


# ---------- facet cleanup ----------

def test_facets_keep_only_well_formed_items_of_the_window():
    window = [turn(5, "What is X?", "X is 5.", document="d.pdf", citations=[{"page": 3}, {"page": 1}])]

    facets = _clean_facets({
        "facts": [
            {"text": "X is 5", "turn": 5},
            {"text": "X is 5", "turn": 5},            # duplicate
            {"text": "ghost", "turn": 99},            # not a turn of this window
            {"text": "", "turn": 5},                  # empty
            "not a dict",
        ],
        "user_facts": [{"text": "My deadline is 15 November", "turn": "5"}, {"text": "x", "turn": 77}],
        "entities": [{"name": "X", "turns": [5, 99]}, {"name": "nowhere", "turns": [99]}],
        "decisions": [{"topic": "skip Y", "status": "closed", "turn": 5}, {"topic": "bad", "turn": 99}],
        "open_questions": ["what is Z?", 7, ""],
    }, window)

    assert [f["text"] for f in facets["facts"]] == ["X is 5"]
    assert facets["facts"][0]["pages"] == [1, 3] and facets["facts"][0]["document"] == "d.pdf"
    assert facets["user_facts"] == [{"text": "My deadline is 15 November", "turn": 5}]
    assert [e["name"] for e in facets["entities"]] == ["X"] and facets["entities"][0]["turns"] == [5]
    assert [d["topic"] for d in facets["decisions"]] == ["skip Y"]
    assert facets["open_questions"] == ["what is Z?"]


def test_asked_facet_comes_from_the_turns_not_the_llm():
    facets = _clean_facets({"asked": [{"turn": 1, "text": "invented"}]}, [turn(4, "Which  case\nis it?", "a")])

    assert facets["asked"] == [{"turn": 4, "text": "Which case is it?"}]


def test_llm_output_of_the_wrong_shape_is_tolerated():
    facets = _clean_facets({"facts": "a string", "entities": None, "decisions": 5}, [turn(1)])

    assert facets["facts"] == [] and facets["entities"] == [] and facets["decisions"] == []


def test_snapshot_text_includes_user_statements_and_questions():
    facets = _clean_facets(
        {"user_facts": [{"text": "The draft deadline is 15 November", "turn": 2}]},
        [turn(2, "My deadline is 15 November. Note it.", "Noted.")],
    )

    text = render_snapshot({"facets": facets})

    assert "User said: The draft deadline is 15 November" in text
    assert "Asked at turn 2: My deadline is 15 November" in text


# ---------- create_snapshot ----------

def test_snapshots_are_hash_chained_immutable_and_indexed(llm, fake_embed, qdrant, chat_factory):
    llm.reply("compress part of a conversation", snapshot_json(facts=[("Fact one", 1)]))
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")])
    turns = session_store.get_turns(user, chat)

    first = create_snapshot(user, chat, turns[:2], [], {"trigger": "t"})
    second = create_snapshot(user, chat, turns[2:], [first], {"trigger": "t"})

    assert first["prev_hash"] is None and second["prev_hash"] == first["state_hash"]
    assert (first["from_turn"], first["to_turn"], second["from_turn"]) == (1, 2, 3)
    assert first["meta"]["compression"] > 0
    assert len(qdrant.collections[settings.SNAPSHOT_COLLECTION]) == 2

    # Written once: saving the same snapshot again must fail.
    with pytest.raises(FileExistsError):
        snapshot_store.save_snapshot(user, chat, first)


def test_state_hash_pins_the_content(llm, fake_embed, qdrant, chat_factory):
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2")])
    turns = session_store.get_turns(user, chat)

    llm.reply("compress part of a conversation", snapshot_json(facts=[("A", 1)]))
    one = create_snapshot(user, chat, turns, [], {})

    other_user, other_chat = chat_factory([("q1", "a1"), ("q2", "a2")])
    llm.rules.clear()
    llm.reply("compress part of a conversation", snapshot_json(facts=[("B", 1)]))
    two = create_snapshot(other_user, other_chat, session_store.get_turns(other_user, other_chat), [], {})

    assert one["state_hash"] != two["state_hash"]


def test_failed_extraction_writes_nothing(llm, fake_embed, qdrant, chat_factory):
    llm.reply("compress part of a conversation", "this is not json")
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2")])

    with pytest.raises(ValueError):
        create_snapshot(user, chat, session_store.get_turns(user, chat), [], {})

    assert snapshot_store.list_snapshots(user, chat) == []


def test_exemplars_show_only_the_llm_facets(llm, fake_embed, qdrant, chat_factory):
    llm.reply("compress part of a conversation", snapshot_json(facts=[("F", 1)]))
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")])
    turns = session_store.get_turns(user, chat)

    first = create_snapshot(user, chat, turns[:2], [], {})
    create_snapshot(user, chat, turns[2:], [first], {})

    prompt = llm.calls_matching("EARLIER SNAPSHOTS")[0]

    assert "asked" not in prompt.split("EARLIER SNAPSHOTS")[1].split("TURNS:")[0]


def test_exemplars_can_be_turned_off(llm, fake_embed, qdrant, chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "SNAPSHOT_EXEMPLARS", 0)
    llm.reply("compress part of a conversation", snapshot_json(facts=[("F", 1)]))
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2"), ("q3", "a3"), ("q4", "a4")])
    turns = session_store.get_turns(user, chat)

    first = create_snapshot(user, chat, turns[:2], [], {})
    second = create_snapshot(user, chat, turns[2:], [first], {})

    assert second["exemplars_used"] == [] and not llm.calls_matching("EARLIER SNAPSHOTS")


# ---------- after_turn ----------

def test_after_turn_snapshots_under_pressure_and_logs_every_turn(llm, fake_embed, qdrant, chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_BUDGET_TOKENS", 100)
    monkeypatch.setattr(settings, "MEMORY_STATE_ENABLED", False)
    llm.reply("compress part of a conversation", snapshot_json(facts=[("F", 1)]))
    llm.reply("EARLIER answers", json.dumps({"conflicts": []}))
    user, chat = chat_factory([("a long question " * 20, "a long answer " * 30)])

    snapshot_engine.after_turn(user, chat)                    # one turn: nothing to compress yet
    assert snapshot_store.list_snapshots(user, chat) == []

    session_store.append_turn(user, chat, "next " * 50, "reply " * 50)
    snapshot_engine.after_turn(user, chat)

    snapshots = snapshot_store.list_snapshots(user, chat)
    log = snapshot_store.read_cqm(user, chat)

    assert len(snapshots) == 1 and (snapshots[0]["from_turn"], snapshots[0]["to_turn"]) == (1, 2)
    assert [r["turn"] for r in log] == [1, 2] and log[1]["trigger"] == "pressure"


def test_after_turn_skips_a_turn_it_already_measured(llm, fake_embed, qdrant, chat_factory):
    llm.reply("EARLIER answers", json.dumps({"conflicts": []}))
    user, chat = chat_factory([("q1", "a1")])

    snapshot_engine.after_turn(user, chat)
    snapshot_engine.after_turn(user, chat)

    assert len(snapshot_store.read_cqm(user, chat)) == 1


def test_a_failed_snapshot_is_retried_on_the_next_turn(llm, fake_embed, qdrant, chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "CQM_FIXED_INTERVAL", 2)
    monkeypatch.setattr(settings, "MEMORY_STATE_ENABLED", False)
    monkeypatch.setattr(settings, "CQM_CHECK_CONTRADICTION", False)
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2")])

    snapshot_engine.after_turn(user, chat)                    # extraction unavailable
    assert snapshot_store.list_snapshots(user, chat) == []
    assert "snapshot_error" in snapshot_store.read_cqm(user, chat)[-1]

    llm.reply("compress part of a conversation", snapshot_json(facts=[("F", 1)]))
    session_store.append_turn(user, chat, "q3", "a3")
    snapshot_engine.after_turn(user, chat)

    assert [(s["from_turn"], s["to_turn"]) for s in snapshot_store.list_snapshots(user, chat)] == [(1, 3)]


# ---------- history shown to the LLM ----------

def test_history_is_the_last_turns_without_snapshots(chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_TURNS", 2)
    monkeypatch.setattr(settings, "HISTORY_MAX_TOKENS", 0)
    user, chat = chat_factory([(f"q{i}", f"a{i}") for i in range(1, 6)])

    assert [t["turn"] for t in history.prompt_history(user, chat)] == [4, 5]


def test_turns_after_the_newest_snapshot_are_added_within_the_token_budget(chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_TURNS", 1)
    monkeypatch.setattr(settings, "HISTORY_MAX_TOKENS", 100)        # 400 chars
    user, chat = chat_factory([(f"q{i}", "x" * 150) for i in range(1, 7)])

    kept = [t["turn"] for t in history.prompt_history(user, chat)]

    # Each turn is 152 characters: the last turn is always there, turn 5 still
    # fits (304), turn 4 would not (456 > 400).
    assert kept == [5, 6]


def test_turns_covered_by_a_snapshot_are_not_repeated(chat_factory, monkeypatch, llm, fake_embed, qdrant):
    monkeypatch.setattr(settings, "HISTORY_TURNS", 1)
    monkeypatch.setattr(settings, "HISTORY_MAX_TOKENS", 10_000)
    llm.reply("compress part of a conversation", snapshot_json(facts=[("F", 1)]))
    user, chat = chat_factory([(f"q{i}", f"a{i}") for i in range(1, 6)])
    create_snapshot(user, chat, session_store.get_turns(user, chat)[:3], [], {})

    assert [t["turn"] for t in history.prompt_history(user, chat)] == [4, 5]


def test_long_answers_are_shortened_in_the_prompt_only(chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "HISTORY_ANSWER_CHARS", 50)
    user, chat = chat_factory([("q", "y" * 500)])

    shown = history.prompt_history(user, chat)[0]["answer"]

    assert shown.endswith("[...]") and len(shown) < 70
    assert len(session_store.get_turns(user, chat)[0]["answer"]) == 500
