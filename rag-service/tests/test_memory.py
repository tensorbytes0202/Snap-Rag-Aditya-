"""Verified reconstruction, conversation state, hand-off and memory recall."""

import json

import pytest

from app.core.config import settings
from app.services import memory_retriever, memory_state, reconstruction, session_store, snapshot_store
from app.services.snapshot_engine import create_snapshot

from conftest import snapshot_json


def snapshot_of(facts=(), user_facts=(), decisions=(), entities=(), from_turn=1, to_turn=2, open_questions=(), asked=()):
    return {
        "state_hash": "h" * 64,
        "seq": 1,
        "from_turn": from_turn,
        "to_turn": to_turn,
        "facets": {
            "facts": [{"text": t, "turn": n, "document": "d.pdf", "pages": [1], "verified": True} for t, n in facts],
            "user_facts": [{"text": t, "turn": n} for t, n in user_facts],
            "entities": [{"name": n, "turns": [1], "documents": ["d.pdf"], "pages": [1]} for n in entities],
            "decisions": [{"topic": t, "status": s, "turn": n} for t, s, n in decisions],
            "open_questions": list(open_questions),
            "asked": [{"turn": n, "text": t} for t, n in asked],
        },
    }


def judge_labels(*labels):
    """A judge reply labelling sentence 1, 2, ... in order."""

    return json.dumps({"checks": [{"i": i, "source": label} for i, label in enumerate(labels, 1)]})


# ---------- reconstruction ladder ----------

def test_snapshot_mode_returns_the_reconstruction_unchecked(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["Router-2 uses 50.50.50.1", "made up"]}))
    user, chat = chat_factory([("q", "a")])

    result = reconstruction.recall(user, chat, snapshot_of(facts=[("Router-2 uses 50.50.50.1", 1)]), "r2?", "snapshot")

    assert [k["text"] for k in result["kept"]] == ["Router-2 uses 50.50.50.1", "made up"]
    assert result["dropped"] == [] and not llm.calls_matching("Check each SENTENCE")


def test_verify_mode_drops_what_no_source_states(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["true one", "invented one"]}))
    llm.reply("Check each SENTENCE", judge_labels("snapshot", "none"))
    user, chat = chat_factory([("q", "a")])

    result = reconstruction.recall(user, chat, snapshot_of(facts=[("true one", 1)]), "q", "verify")

    assert [k["text"] for k in result["kept"]] == ["true one"]
    assert result["dropped"] == ["invented one"]
    assert not llm.calls_matching('SOURCE "archive"')                 # no archive tier in this mode


def test_archive_mode_rescues_a_doubtful_sentence_from_the_chat_record(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["in snapshot", "only in the archive", "nowhere"]}))
    llm.reply('SOURCE "snapshot"', judge_labels("snapshot", "none", "none"))
    llm.reply('SOURCE "archive"', judge_labels("archive", "none"))
    user, chat = chat_factory([("q", "a")])

    result = reconstruction.recall(user, chat, snapshot_of(facts=[("in snapshot", 1)]), "q", "archive")

    assert [(k["text"], k["source"]) for k in result["kept"]] == [("in snapshot", "snapshot"), ("only in the archive", "archive")]
    assert result["dropped"] == ["nowhere"]


def test_when_the_check_cannot_run_nothing_counts_as_verified(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["a sentence"]}))      # the judge is unreachable
    user, chat = chat_factory([("q", "a")])

    result = reconstruction.recall(user, chat, snapshot_of(facts=[("a sentence", 1)]), "q", "archive")

    assert result["kept"] == [] and result["verification_failed"] is True


def test_a_judge_that_skips_a_sentence_can_only_drop_it(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["one", "two"]}))
    llm.reply("Check each SENTENCE", json.dumps({"checks": [{"i": 1, "source": "snapshot"}]}))
    user, chat = chat_factory([("q", "a")])

    result = reconstruction.recall(user, chat, snapshot_of(facts=[("one", 1)]), "q", "verify")

    assert [k["text"] for k in result["kept"]] == ["one"]


def test_a_judge_inventing_a_source_label_is_ignored(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["one"]}))
    llm.reply("Check each SENTENCE", judge_labels("[turn 3]"))
    user, chat = chat_factory([("q", "a")])

    assert reconstruction.recall(user, chat, snapshot_of(facts=[("one", 1)]), "q", "verify")["kept"] == []


def test_an_unknown_mode_is_rejected(chat_factory):
    user, chat = chat_factory([("q", "a")])

    with pytest.raises(ValueError):
        reconstruction.recall(user, chat, snapshot_of(), "q", "magic")


def test_the_archive_is_cut_to_the_snapshots_turn_range(llm, chat_factory):
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["x"]}))
    llm.reply('SOURCE "snapshot"', judge_labels("none"))
    llm.reply('SOURCE "archive"', judge_labels("archive"))
    user, chat = chat_factory([(f"question {i}", f"answer {i}") for i in range(1, 6)])

    reconstruction.recall(user, chat, snapshot_of(from_turn=2, to_turn=3), "q", "archive")

    archive_prompt = llm.calls_matching('SOURCE "archive"')[0]

    assert "question 2" in archive_prompt and "question 3" in archive_prompt
    assert "question 1" not in archive_prompt and "question 4" not in archive_prompt


# ---------- snapshot recall (fusion) ----------

def test_memory_weight_rises_with_the_turn_and_can_be_fixed(monkeypatch):
    weights = [memory_retriever.memory_weight(t) for t in (1, 8, 20, 50)]

    assert weights == sorted(weights) and weights[1] == pytest.approx(0.5)
    assert memory_retriever.memory_weight(1) == pytest.approx(0.36, abs=0.01)

    monkeypatch.setattr(settings, "LAMBDA_FIXED", 0.9)
    assert memory_retriever.memory_weight(1) == 0.9


def test_recall_is_off_in_the_baseline_mode(monkeypatch, chat_factory):
    monkeypatch.setattr(settings, "MEMORY_RECALL_MODE", "off")
    user, chat = chat_factory([("q", "a")])

    assert memory_retriever.recall_memory(user, chat, "q", "d.pdf") is None
    assert memory_retriever.finalize_memory(None, "ctx") is None
    assert memory_retriever.memory_block(None) == ""


def test_merge_leaves_out_other_documents_and_keeps_the_conversation():
    mine = snapshot_of(facts=[("about d", 1)], user_facts=[("deadline", 2)], decisions=[("skip", "closed", 2)])
    other = snapshot_of(facts=[("about e", 1)])
    other["facets"]["facts"][0]["document"] = "e.pdf"

    merged = memory_retriever.merge_snapshots([mine, other], "d.pdf")["facets"]

    assert [f["text"] for f in merged["facts"]] == ["about d"]
    assert [u["text"] for u in merged["user_facts"]] == ["deadline"] and merged["decisions"]


def test_merge_can_leave_the_conversation_to_the_state_block():
    merged = memory_retriever.merge_snapshots(
        [snapshot_of(facts=[("f", 1)], user_facts=[("u", 2)], decisions=[("d", "closed", 2)], open_questions=["q"])],
        "d.pdf", with_conversation=False,
    )["facets"]

    assert merged["facts"] and not merged["user_facts"] and not merged["decisions"] and not merged["open_questions"]


def test_finalize_packs_by_lambda_budget_and_drops_duplicates(monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_CONTEXT_TOKENS", 100)
    memory = {
        "lambda": 0.5,                                                   # budget: 50 tokens = 200 chars
        "candidates": [{"text": "A" * 80, "source": "snapshot"}, {"text": "a" * 80, "source": "snapshot"},
                       {"text": "already in the passages", "source": "snapshot"},
                       {"text": "B" * 160, "source": "snapshot"}, {"text": "C" * 10, "source": "snapshot"}],
    }

    done = memory_retriever.finalize_memory(memory, "context ... Already in the passages. ...")

    # The case-insensitive duplicate and the sentence the document already has are
    # skipped; "B"*160 (40 tokens) would overshoot the 50-token budget (20 + 40),
    # which ends the packing.
    assert [s["text"] for s in done["sentences"]] == ["A" * 80]
    assert done["tokens"] == 20 and "candidates" not in done


def test_recall_returns_the_state_even_when_no_snapshot_matches(llm, fake_embed, qdrant, chat_factory):
    user, chat = chat_factory([("q1", "a1"), ("q2", "a2")])
    snapshot_store.append_state_version(user, chat, {"version": 1, "covered": [], "items": [
        {"kind": "user_fact", "text": "The draft deadline is 15 November", "turn": 1},
    ]})

    memory = memory_retriever.recall_memory(user, chat, "what is my deadline?", "d.pdf")

    assert memory["state_items"] == 1 and memory["snapshots"] == []
    assert "The draft deadline is 15 November" in memory_retriever.memory_block(memory)


def test_state_can_be_switched_off_for_the_ablation(fake_embed, qdrant, chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_STATE_ENABLED", False)
    user, chat = chat_factory([("q1", "a1")])
    snapshot_store.append_state_version(user, chat, {"version": 1, "covered": [], "items": [
        {"kind": "user_fact", "text": "x", "turn": 1}]})

    assert memory_retriever.recall_memory(user, chat, "q", "d.pdf")["state_text"] == ""


def test_recall_end_to_end_reconstructs_and_verifies_matching_snapshots(llm, fake_embed, qdrant, chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_STATE_ENABLED", False)
    llm.reply("compress part of a conversation",
              snapshot_json(facts=[("Router-2 routes 10.0.0.0/8 via 50.50.50.1", 1)], entities=[("Router-2", [1])]))
    llm.reply("rebuild conversation memory", json.dumps({"sentences": ["Router-2 routes 10.0.0.0/8 via 50.50.50.1", "Router-9 exists"]}))
    llm.reply("Check each SENTENCE", judge_labels("snapshot", "none"))
    user, chat = chat_factory([("What does Router-2 route?", "Router-2 routes 10.0.0.0/8 via 50.50.50.1")] * 2, document="d.pdf")
    create_snapshot(user, chat, session_store.get_turns(user, chat), [], {})

    memory = memory_retriever.recall_memory(user, chat, "What does Router-2 route?", "d.pdf")
    memory = memory_retriever.finalize_memory(memory, "unrelated document text")

    assert [s["text"] for s in memory["sentences"]] == ["Router-2 routes 10.0.0.0/8 via 50.50.50.1"]
    assert memory["dropped_unverified"] == 1 and memory["snapshots"]
    assert "VERIFIED MEMORY" in memory_retriever.memory_block(memory)


def test_a_failing_recall_never_breaks_answering(llm, fake_embed, qdrant, chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "MEMORY_STATE_ENABLED", False)
    llm.reply("compress part of a conversation", snapshot_json(facts=[("Router-2 fact", 1)]))
    user, chat = chat_factory([("Router-2 question", "Router-2 fact")] * 2, document="d.pdf")
    create_snapshot(user, chat, session_store.get_turns(user, chat), [], {})

    memory = memory_retriever.recall_memory(user, chat, "Router-2 question", "d.pdf")        # reconstruction call fails

    assert memory["sentences"] == [] and "error" in memory


# ---------- conversation state ----------

STATEMENT = "The user's draft deadline is 15 November"


def make_state_chat(chat_factory, llm, snapshot_reply, questions=None):
    questions = questions or ["My deadline is 15 November. Note it.", "Explain section 26."]
    llm.reply("compress part of a conversation", snapshot_reply)
    user, chat = chat_factory([(q, "answer") for q in questions])
    return user, chat, [create_snapshot(user, chat, session_store.get_turns(user, chat), [], {})]


def test_verified_user_statements_enter_the_state(llm, fake_embed, qdrant, chat_factory):
    user, chat, _ = make_state_chat(chat_factory, llm, snapshot_json(
        user_facts=[(STATEMENT, 1)], decisions=[("Report covers only section 26", "closed", 2)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages", "user_messages"))

    version = memory_state.update_state(user, chat)

    assert version["version"] == 1
    assert {i["text"] for i in version["items"]} == {STATEMENT, "Report covers only section 26"}
    assert memory_state.current_state(user, chat) == version["items"]


def test_statements_the_user_never_made_are_rejected(llm, fake_embed, qdrant, chat_factory):
    user, chat, _ = make_state_chat(chat_factory, llm, snapshot_json(
        user_facts=[(STATEMENT, 1), ("The user's advisor is Dr. Rao", 2)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages", "none"))

    version = memory_state.update_state(user, chat)

    assert [i["text"] for i in version["items"]] == [STATEMENT]
    assert [r["text"] for r in version["rejected"]] == ["The user's advisor is Dr. Rao"]


def test_the_check_only_sees_the_users_own_messages(llm, fake_embed, qdrant, chat_factory):
    user, chat, _ = make_state_chat(chat_factory, llm, snapshot_json(user_facts=[(STATEMENT, 1)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))

    memory_state.update_state(user, chat)

    prompt = llm.calls_matching("Check each SENTENCE")[0]

    assert "My deadline is 15 November" in prompt and "answer" not in prompt.split("SENTENCES:")[0].split("user_messages")[1]


def test_when_the_check_is_unavailable_the_snapshot_stays_pending(llm, fake_embed, qdrant, chat_factory):
    user, chat, _ = make_state_chat(chat_factory, llm, snapshot_json(user_facts=[(STATEMENT, 1)]))

    assert memory_state.update_state(user, chat) is None                      # the judge is unreachable
    assert memory_state.current_state(user, chat) == []

    llm.reply("Check each SENTENCE", judge_labels("user_messages"))
    version = memory_state.update_state(user, chat)                           # the next update catches up

    assert [i["text"] for i in version["items"]] == [STATEMENT]


def test_a_snapshot_is_only_folded_in_once(llm, fake_embed, qdrant, chat_factory):
    user, chat, _ = make_state_chat(chat_factory, llm, snapshot_json(user_facts=[(STATEMENT, 1)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))

    memory_state.update_state(user, chat)

    assert memory_state.update_state(user, chat) is None
    assert len(snapshot_store.read_state_versions(user, chat)) == 1


def test_a_new_value_supersedes_the_old_one(llm, fake_embed, qdrant, chat_factory):
    user, chat, first = make_state_chat(chat_factory, llm, snapshot_json(user_facts=[(STATEMENT, 1)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))
    memory_state.update_state(user, chat)

    session_store.append_turn(user, chat, "Actually my deadline moved to 30 November.", "Noted.")
    session_store.append_turn(user, chat, "Explain section 27.", "answer")
    llm.rules.clear()
    llm.reply("compress part of a conversation", snapshot_json(user_facts=[("The user's draft deadline is 30 November", 3)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))
    llm.reply("remembered from a conversation", json.dumps({"replaced": [{"old": 1, "by": 1}], "answered": []}))
    turns = session_store.get_turns(user, chat)
    create_snapshot(user, chat, turns[2:], first, {})

    version = memory_state.update_state(user, chat)

    assert [i["text"] for i in version["items"]] == ["The user's draft deadline is 30 November"]
    assert version["superseded"][0]["text"] == STATEMENT and version["version"] == 2


def test_consolidation_only_points_at_items_and_never_rewrites_them(llm, fake_embed, qdrant, chat_factory):
    old = [{"kind": "user_fact", "text": "old", "turn": 1}]
    new = [{"kind": "user_fact", "text": "new", "turn": 2}]
    llm.reply("remembered from a conversation", json.dumps({"replaced": [{"old": 9, "by": 1}, {"old": 1, "by": 7}, "junk"], "answered": ["x"]}))

    assert memory_state._consolidate(old, new) == (set(), set())              # out-of-range pointers are ignored


def test_an_unreachable_consolidation_keeps_every_item(llm):
    old = [{"kind": "user_fact", "text": "old", "turn": 1}]

    assert memory_state._consolidate(old, [{"kind": "user_fact", "text": "new", "turn": 2}]) == (set(), set())


def test_answered_open_questions_are_closed(llm, fake_embed, qdrant, chat_factory):
    user, chat, first = make_state_chat(chat_factory, llm, snapshot_json(open_questions=["Which sections do I cover?"]),
                                        ["Which sections do I cover? Not sure yet.", "Explain section 26."])
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))
    memory_state.update_state(user, chat)

    session_store.append_turn(user, chat, "I will cover sections 26 and 27.", "Noted.")
    session_store.append_turn(user, chat, "Explain section 27.", "answer")
    llm.rules.clear()
    llm.reply("compress part of a conversation", snapshot_json(decisions=[("Cover sections 26 and 27", "closed", 3)]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))
    llm.reply("remembered from a conversation", json.dumps({"replaced": [], "answered": [1]}))
    create_snapshot(user, chat, session_store.get_turns(user, chat)[2:], first, {})

    kinds = [i["kind"] for i in memory_state.update_state(user, chat)["items"]]

    assert kinds == ["decision"]


def test_state_block_fits_its_budget_and_keeps_the_newest(chat_factory):
    user, chat = chat_factory([("q", "a")])
    items = [{"kind": "user_fact", "text": f"fact number {i} " + "x" * 40, "turn": i} for i in range(1, 30)]
    snapshot_store.append_state_version(user, chat, {"version": 1, "covered": [], "items": items})

    block = memory_state.state_block(user, chat, max_tokens=50)         # 200 characters

    # Each line is ~80 characters: two fit in 200, the third would not.
    assert "fact number 29" in block and "fact number 28" in block and "fact number 27" not in block
    assert block.index("fact number 28") < block.index("fact number 29")         # shown oldest first


def test_no_state_no_block(chat_factory):
    user, chat = chat_factory([("q", "a")])

    assert memory_state.state_block(user, chat) == ""


# ---------- hand-off ----------

def test_handoff_lists_the_state_and_where_the_chat_stands(llm, fake_embed, qdrant, chat_factory):
    user, chat, _ = make_state_chat(chat_factory, llm, snapshot_json(
        user_facts=[(STATEMENT, 1)], entities=[("Bate v. Third Ave. R. Co.", [2])]))
    llm.reply("Check each SENTENCE", judge_labels("user_messages"))
    memory_state.update_state(user, chat)

    text = memory_state.build_handoff(user, chat)

    assert STATEMENT in text and "Bate v. Third Ave. R. Co." in text
    assert "Last question (turn 2): Explain section 26." in text


def test_handoff_is_capped(chat_factory, monkeypatch):
    monkeypatch.setattr(settings, "HANDOFF_MAX_TOKENS", 20)
    user, chat = chat_factory([("question " * 50, "a")] * 3)

    assert len(memory_state.build_handoff(user, chat)) <= 80


def test_only_a_resumed_session_gets_the_handoff(chat_factory, monkeypatch):
    user, chat = chat_factory([("q", "a")])
    last = session_store.get_turns(user, chat)[-1]["ts"]

    assert not memory_state.is_resumed(user, chat, now=last + 60)
    assert memory_state.is_resumed(user, chat, now=last + settings.SESSION_GAP_SECONDS + 1)
    assert memory_state.handoff_block(user, chat) == ""                    # just asked: not a resume

    monkeypatch.setattr(settings, "SESSION_GAP_SECONDS", -1)
    assert "HAND-OFF FROM THE EARLIER SESSION" in memory_state.handoff_block(user, chat)


def test_an_empty_chat_has_no_handoff(chat_factory):
    user, chat = chat_factory([])

    assert memory_state.build_handoff(user, chat) == "" and not memory_state.is_resumed(user, chat)
