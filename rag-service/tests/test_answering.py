"""AB-RAG (confidence signals, batches), answer checking and the query flow around them."""

import json

import pytest

from app.core.config import settings
from app.services import abrag, answering, query_planner, query_router, query_service, verification_service
from app.services.abrag import Attempt, run_batches, score_answer, similarity_signal, spread_signal
from app.services.query_runner import Retrieval


def retrieval(passages=8, limit=3, query_type="EXPLANATION", rerank=(3.0, 1.0, 0.0, -1.0), plan=None):
    """A Retrieval holding `passages` ranked chunks, of which `limit` are read."""

    chunks = [{"page": 10 + i, "document": "d.pdf", "text": f"passage {i} about negligence", "parent_text": "",
               "rerank_score": rerank[i % len(rerank)]} for i in range(passages)]

    found = Retrieval(["q"], query_type, dict(query_router.get_retrieval_config(query_type)),
                      ranked_chunks=chunks, results=chunks, limit=limit, plan=plan or {"id": 1})
    found.sources = [{"document": "d.pdf", "page": c["page"]} for c in found.top_chunks]

    return found


def attempt(confidence, supported=True, strategy="initial", **kwargs):
    return Attempt(strategy, "answer", {"supported": supported, "confidence": confidence, "reason": "r",
                                        "unsupported_claims": ["claim one"]}, confidence, retrieval(**kwargs))


# ---------- signals ----------

def test_spread_is_the_scaled_standard_deviation_of_the_reranker_scores():
    assert spread_signal([4.0, -2.0]) == 1.0                 # a clear winner among the passages
    assert spread_signal([1.0, 1.0, 1.0]) == 0.0             # nothing separated
    assert 0 < spread_signal([2.0, 1.0, 0.0]) < 1
    assert spread_signal([2.0]) == 0.5                       # one passage says nothing about spread


def test_similarity_is_the_closest_evidence_scaled(fake_embed):
    close = similarity_signal("proximate cause negligence rule", ["unrelated words here", "proximate cause negligence rule"])
    far = similarity_signal("proximate cause negligence rule", ["zebra quantum banana", "volcano tuesday"])

    assert close == 1.0 and far == 0.0
    assert similarity_signal("", ["x"]) == 0.0 and similarity_signal("x", []) == 0.0


def test_confidence_is_the_weighted_mean_of_the_three_signals(fake_embed):
    found = retrieval(rerank=(4.0, -2.0), limit=2)
    verification = {"supported": True, "confidence": 0.8}

    confidence, signals = score_answer(found, "passage 0 about negligence", verification)

    assert signals["model"] == 0.8 and signals["spread"] == 1.0 and signals["similarity"] > 0.5
    assert confidence == round(0.5 * 0.8 + 0.25 * signals["similarity"] + 0.25 * 1.0, 2)


def test_memory_counts_as_evidence(fake_embed):
    found = retrieval()
    answer = "your deadline is fifteen november"
    verification = {"supported": True, "confidence": 0.9}

    without = score_answer(found, answer, verification)[1]["similarity"]
    with_memory = score_answer(found, answer, verification, ["your deadline is fifteen november"])[1]["similarity"]

    assert with_memory == 1.0 and without < with_memory


def test_an_unsupported_answer_can_never_reach_the_threshold(fake_embed):
    confidence, _ = score_answer(retrieval(rerank=(4.0, -2.0)), "passage 0 about negligence",
                                 {"supported": False, "confidence": 1.0})

    assert confidence == settings.STOP_CONFIDENCE - 0.01


def test_conversation_answers_rest_on_the_checker_alone(fake_embed):
    confidence, signals = score_answer(retrieval(query_type="CONVERSATION"), "anything", {"supported": True, "confidence": 0.95})

    assert confidence == 0.95 and signals == {"model": 0.95}


def test_garbage_checker_confidence_counts_as_zero(fake_embed):
    assert score_answer(retrieval(query_type="CONVERSATION"), "a", {"supported": True, "confidence": "high"})[0] == 0.0


# ---------- intent budgets and the next batch ----------

def test_every_intent_has_a_first_batch_and_a_budget():
    for name, config in query_router.CONFIGS.items():
        assert config["passages"] >= 1 and config["batches"] >= 0, name

    assert query_router.CONFIGS["CONVERSATION"]["batches"] == 0
    assert query_router.CONFIGS["SUMMARY"]["passages"] > query_router.CONFIGS["DEFINITION"]["passages"]
    assert query_router.get_retrieval_config("NONSENSE") == query_router.CONFIGS["EXPLANATION"]


def test_the_next_batch_reads_deeper_into_the_same_ranking_without_mutating_it():
    first = retrieval(passages=8, limit=3)

    second = first.deeper(2)

    assert (first.limit, second.limit) == (3, 5)
    assert len(first.sources) == 3 and len(second.sources) == 5
    assert second.ranked_chunks is first.ranked_chunks


def test_the_ranking_runs_out():
    assert retrieval(passages=3, limit=3).deeper(2) is None
    assert retrieval(passages=8, limit=18).deeper(2) is None                       # MAX_PASSAGES
    assert retrieval(passages=8, limit=7).deeper(5).limit == 8


# ---------- the loop ----------

class Recorder:

    def __init__(self, results, fresh=None):
        self.results = list(results)
        self.fresh = fresh
        self.retrievals, self.strategies, self.limits = [], [], []

    def retrieve(self, **overrides):
        self.retrievals.append(overrides)
        return self.fresh or retrieval(passages=8, limit=5)

    def attempt_from(self, found, strategy):
        self.strategies.append(strategy)
        self.limits.append(found.limit)
        return self.results.pop(0)


def run(first, recorder, question="q"):
    return run_batches(question, first, recorder.retrieve, recorder.attempt_from)


def test_a_confident_first_answer_reads_no_more(llm):
    recorder = Recorder([])

    best, log = run(attempt(0.95), recorder)

    assert best.confidence == 0.95 and recorder.strategies == [] and len(log) == 1


def test_a_weak_answer_reads_the_next_batch_for_free(llm):
    recorder = Recorder([attempt(0.9, strategy="deeper")])

    best, log = run(attempt(0.5), recorder)

    assert recorder.retrievals == [] and recorder.strategies == ["deeper"] and recorder.limits == [5]    # 3 + max(2, 3 // 2)
    assert best.confidence == 0.9 and [entry["strategy"] for entry in log] == ["initial", "deeper"]


def test_batches_stop_as_soon_as_the_answer_is_confident(llm):
    recorder = Recorder([attempt(0.6), attempt(0.85), attempt(0.99)])

    best, _ = run(attempt(0.5), recorder)

    assert recorder.strategies == ["deeper", "deeper"] and best.confidence == 0.85


def test_a_fresh_wider_search_runs_only_when_the_ranking_is_used_up(llm):
    llm.reply("found problems with the answer", json.dumps({"queries": ["last clear chance"]}))
    recorder = Recorder([attempt(0.9, strategy="fresh")])

    best, _ = run(attempt(0.5, passages=3, limit=3), recorder)

    assert recorder.retrievals == [{"plan": {"id": 1}, "extra_queries": ["last clear chance"], "wide": True}]
    assert recorder.strategies == ["fresh"] and best.confidence == 0.9


def test_the_best_attempt_wins_not_the_latest(llm):
    recorder = Recorder([attempt(0.4), attempt(0.5)])

    best, log = run(attempt(0.7), recorder)

    assert best.confidence == 0.7 and len(log) == 3


def test_the_budget_is_the_intents_and_capped_by_the_setting(llm, monkeypatch):
    recorder = Recorder([attempt(0.1)] * 5)

    run(attempt(0.1, query_type="DEFINITION"), recorder)                       # DEFINITION: 1 extra batch
    assert len(recorder.strategies) == 1

    recorder = Recorder([attempt(0.1)] * 5)
    run(attempt(0.1, query_type="SUMMARY"), recorder)                          # SUMMARY: 3
    assert len(recorder.strategies) == 3

    monkeypatch.setattr(settings, "MAX_EXTRA_BATCHES", 2)
    recorder = Recorder([attempt(0.1)] * 5)
    run(attempt(0.1, query_type="SUMMARY"), recorder)
    assert len(recorder.strategies) == 2


def test_no_extra_batches_is_fixed_k_rag(llm, monkeypatch):
    monkeypatch.setattr(settings, "MAX_EXTRA_BATCHES", 0)
    recorder = Recorder([])

    run(attempt(0.1, supported=False), recorder)

    assert recorder.strategies == [] and recorder.retrievals == []


def test_conversation_questions_are_never_searched_again(llm):
    recorder = Recorder([])

    best, _ = run(attempt(0.3, supported=False, query_type="CONVERSATION"), recorder)

    assert recorder.strategies == [] and best.confidence == 0.3


def test_a_fresh_search_that_finds_nothing_ends_the_loop(llm):
    llm.reply("found problems with the answer", json.dumps({"queries": ["x"]}))
    recorder = Recorder([], fresh=retrieval(passages=0))

    best, log = run(attempt(0.5, passages=3, limit=3), recorder)

    assert len(recorder.retrievals) == 1 and len(log) == 1 and best.confidence == 0.5


def test_refinement_falls_back_to_the_unsupported_claims(llm):
    weak = attempt(0.6)
    weak.verification["unsupported_claims"] = ["the operator may presume the walker will move away", 7]

    assert abrag.refine_queries("q", weak) == ["the operator may presume the walker will move away"]


# ---------- verifier ----------

def test_an_entity_mismatch_makes_the_answer_unsupported(llm):
    llm.reply("expert RAG verification", json.dumps({"supported": True, "confidence": 0.99, "entity_mismatch": True}))

    assert verification_service.verify_answer("What did Templeton say?", "Romney Marsh ...", "ctx")["supported"] is False


def test_unparseable_verification_counts_as_unsupported(llm):
    llm.reply("expert RAG verification", "not json at all")

    assert verification_service.verify_answer("q", "a", "ctx")["supported"] is False


def test_a_verification_that_is_valid_json_but_not_an_object_is_unsupported(llm):
    llm.reply("expert RAG verification", "[1, 2]")

    assert verification_service.verify_answer("q", "a", "ctx")["supported"] is False


# ---------- presenting ----------

def test_a_supported_answer_is_shown_as_generated():
    assert answering.present_answer(attempt(0.9)) == "answer"


def test_an_unsupported_answer_lists_what_could_not_be_confirmed(monkeypatch):
    monkeypatch.setattr(settings, "DROP_UNVERIFIED_ANSWERS", False)

    shown = answering.present_answer(attempt(0.3, supported=False))

    assert shown.startswith("answer") and "- claim one" in shown


def test_unsupported_answers_can_be_replaced_outright(monkeypatch):
    monkeypatch.setattr(settings, "DROP_UNVERIFIED_ANSWERS", True)

    assert answering.present_answer(attempt(0.3, supported=False)) == answering.UNVERIFIED_ANSWER


def test_memory_texts_are_the_sentences_and_the_state(chat_factory):
    memory = {"sentences": [{"text": "a"}, {"text": "b"}], "state_text": "STATE"}

    assert answering.memory_texts(memory) == ["a", "b", "STATE"]
    assert answering.memory_texts(None) == [] and answering.memory_texts({"sentences": []}) == []


# ---------- query type guess ----------

@pytest.mark.parametrize("question", [
    "Which sections did I say I would NOT cover in my report?",
    "What is my draft deadline?",
    "List all the cases you have mentioned so far, with their PDF pages.",
    "Which case did I ask about at turn 5?",
    "What did you say about res ipsa earlier?",
    "Note this for later: the draft deadline for my torts coursework is 15 November.",
    "My advisor is Dr. Meera Rao, and she wants Bluebook-style citations. Note that.",
])
def test_conversation_questions_are_recognised_without_the_llm(question):
    assert query_planner.guess_query_type(question) == "CONVERSATION"


@pytest.mark.parametrize("question", [
    "What does the book say about res ipsa loquitur?",
    "Explain section 26 proximate cause in detail.",
    "Define proximate cause in one sentence.",
])
def test_document_questions_are_not_taken_for_conversation(question):
    assert query_planner.guess_query_type(question) == "EXPLANATION"


def test_the_planner_keeps_the_heuristic_type_when_the_llm_is_down(llm):
    assert query_planner.plan_query("What is my draft deadline?")["query_type"] == "CONVERSATION"


def test_the_planner_accepts_the_conversation_type_from_the_llm(llm):
    llm.reply("You plan retrieval", json.dumps({"query_type": "conversation", "subqueries": ["deadline"]}))

    assert query_planner.plan_query("whatever") == {"query_type": "CONVERSATION", "subqueries": ["deadline"]}


# ---------- process_query ----------

class Pipeline:
    """process_query with retrieval stubbed; the LLM replies are scripted."""

    def __init__(self):
        self.found = retrieval(passages=8, limit=3)
        self.retrievals, self.memory, self.cache = [], None, {}


@pytest.fixture
def pipeline(monkeypatch, llm, fake_embed):

    state = Pipeline()

    def run_retrieval(question, active_document=None, **overrides):
        state.retrievals.append(overrides)
        return state.found

    monkeypatch.setattr(query_service, "run_retrieval", run_retrieval)
    monkeypatch.setattr(query_service, "recall_memory", lambda *a: state.memory)
    monkeypatch.setattr(query_service, "get_cache", lambda key: state.cache.get(key))
    monkeypatch.setattr(query_service, "set_cache", lambda key, value: state.cache.update({key: value}))

    llm.reply("expert RAG verification", json.dumps({"supported": True, "confidence": 0.95}))
    llm.reply("Generate exactly 3 follow-up", "one?\ntwo?\nthree?")
    llm.reply("Advanced RAG Assistant", "passage 0 about negligence")

    return state


def test_a_confident_answer_reads_one_batch(pipeline, llm):
    response = query_service.process_query("What is X?", "d.pdf")

    assert response["answer"] == "passage 0 about negligence" and response["followup_questions"] == ["one?", "two?", "three?"]
    assert response["confidence"] >= settings.STOP_CONFIDENCE and [a["strategy"] for a in response["attempts"]] == ["initial"]
    assert set(response["signals"]) == {"model", "similarity", "spread"}
    assert len(llm.calls_matching("Advanced RAG Assistant")) == 1


def test_a_weak_answer_reads_a_deeper_batch_and_the_better_one_wins(pipeline, llm):
    replies = iter([{"supported": False, "confidence": 0.4, "unsupported_claims": ["c"]},
                    {"supported": True, "confidence": 0.95}])
    llm.rules = [r for r in llm.rules if r[0] != "expert RAG verification"]
    llm.reply("expert RAG verification", lambda prompt: json.dumps(next(replies)))

    response = query_service.process_query("What is X?", "d.pdf")

    assert [a["strategy"] for a in response["attempts"]] == ["initial", "deeper"]
    assert [a["passages"] for a in response["attempts"]] == [3, 5]
    assert response["verification"]["supported"] is True
    assert pipeline.retrievals == [{}]                           # no fresh search: the ranking had more


def test_nothing_found_returns_the_failure_response(pipeline):
    pipeline.found = retrieval(passages=0)

    assert query_service.process_query("q", "d.pdf")["verification"] == "FAILED"


def test_memory_is_fused_into_the_context_the_llm_reads(pipeline, llm):
    pipeline.memory = {"lambda": 0.5, "snapshots": [], "sentences": [], "tokens": 0, "state_items": 1, "handoff": False,
                       "state_text": "\n\nCONVERSATION STATE (x):\n- User stated (turn 1): deadline is 15 November",
                       "candidates": [{"text": "Router-2 uses 50.50.50.1", "source": "snapshot"}]}

    response = query_service.process_query("q", "d.pdf", session=("u", "c"))

    prompt = llm.calls_matching("Advanced RAG Assistant")[0]

    assert "deadline is 15 November" in prompt and "Router-2 uses 50.50.50.1" in prompt
    assert response["retrieved_memory"]["sentences"][0]["text"] == "Router-2 uses 50.50.50.1"


def test_chat_queries_are_not_cached_but_plain_ones_are(pipeline):
    query_service.process_query("Same question", "d.pdf")
    query_service.process_query("Same question", "d.pdf")
    assert len(pipeline.retrievals) == 1                          # the second came from the cache

    query_service.process_query("Same question", "d.pdf", session=("u", "c"))
    query_service.process_query("Same question", "d.pdf", session=("u", "c"))
    assert len(pipeline.retrievals) == 3


def test_an_llm_outage_while_checking_surfaces_to_the_route(pipeline, llm):
    llm.rules = [r for r in llm.rules if r[0] != "expert RAG verification"]

    from app.core.exceptions import LLMProviderError

    with pytest.raises(LLMProviderError):
        query_service.process_query("q", "d.pdf")
