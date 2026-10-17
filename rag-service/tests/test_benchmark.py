"""The long-conversation benchmark: script, scoring, report and runner."""

import pytest

from benchmarks.longconv import report, run, scoring, variants
from benchmarks.longconv.dryrun import DryRunSystem
from benchmarks.longconv.script import _scaled_buckets, build_script, load_spec, probes

SPEC = load_spec()
REFUSAL = SPEC["refusal_pattern"]


# ---------- script ----------

def test_the_script_has_every_turn_once_in_order():
    script = build_script(SPEC, 100, seed=0)

    assert [t["turn"] for t in script] == list(range(1, 101))
    assert all(t["text"] for t in script)


def test_the_script_contains_every_planted_fact_and_probe():
    script = build_script(SPEC, 100, seed=0)
    kinds = [t["kind"] for t in script]

    assert kinds.count("plant") == 14 and kinds.count("correction") == 2
    assert len(probes(script)) == 24
    assert {p["probe_kind"] for p in probes(script)} == {"recall", "supersede", "reference", "absent", "trap"}
    assert len({t["text"] for t in script}) == 100                  # no turn is asked twice


def test_the_same_seed_gives_the_same_script_and_another_seed_a_different_one():
    assert build_script(SPEC, 100, 3) == build_script(SPEC, 100, 3)
    assert build_script(SPEC, 100, 3) != build_script(SPEC, 100, 4)


@pytest.mark.parametrize("seed", range(6))
def test_probes_come_after_their_source_at_the_stated_distance(seed):
    script = build_script(SPEC, 100, seed)
    buckets = _scaled_buckets(SPEC, 100)

    for probe in probes(script):

        if probe["probe_kind"] in ("absent", "trap"):
            assert probe["source_turns"] == [] and probe["bucket"] is None
            continue

        assert all(source < probe["turn"] for source in probe["source_turns"])
        assert probe["distance"] == probe["turn"] - max(probe["source_turns"])

        low, high = buckets[probe["bucket"]]
        assert low <= probe["distance"] <= high


def test_a_correction_comes_between_its_plant_and_probe():
    script = build_script(SPEC, 100, 1)
    by_id = {}

    for t in script:
        if t["kind"] in ("plant", "correction"):
            by_id.setdefault(t["plant_id"], {})[t["kind"]] = t["turn"]

    for probe in probes(script):
        if probe["probe_kind"] == "supersede":
            first, second = probe["source_turns"]
            assert first < second < probe["turn"]
            assert by_id[probe["probe_id"]] == {"plant": first, "correction": second}


def test_a_reference_probe_names_the_turn_of_a_document_question_with_that_topic():
    script = build_script(SPEC, 100, 2)

    for probe in probes(script):
        if probe["probe_kind"] == "reference":
            source = script[probe["source_turns"][0] - 1]
            assert source["kind"] == "doc" and f"turn {source['turn']}?" in probe["text"]
            assert probe["expect_all"] == [source["keyword"]]


def test_reference_keywords_are_not_planted_words():
    plant_text = " ".join(t["text"].lower() for t in build_script(SPEC, 100, 0) if t["kind"] in ("plant", "correction"))

    for probe in probes(build_script(SPEC, 100, 0)):
        if probe["probe_kind"] == "reference":
            assert probe["expect_all"][0].lower() not in plant_text


@pytest.mark.parametrize("turns", [80, 100, 150])
def test_scripts_of_other_lengths_build(turns):
    script = build_script(SPEC, turns, 0)

    assert len(script) == turns and len(probes(script)) == 24


@pytest.mark.parametrize("turns", [25, 40])
def test_a_script_too_short_for_the_probes_is_refused(turns):
    with pytest.raises(ValueError):
        build_script(SPEC, turns, 0)


# ---------- scoring ----------

def test_every_gold_answer_scores_correct():
    for seed in range(4):
        for probe in probes(build_script(SPEC, 100, seed)):
            graded = scoring.score_probe(probe, probe["gold"], REFUSAL)
            assert graded["correct"], (probe["probe_id"], probe["gold"], graded)


def test_a_forgotten_fact_scores_wrong_and_a_declined_absent_question_scores_right():
    script = build_script(SPEC, 100, 0)
    forgot = "Insufficient information found."

    for probe in probes(script):
        graded = scoring.score_probe(probe, forgot, REFUSAL)

        if probe["probe_kind"] in ("recall", "supersede", "reference"):
            assert not graded["correct"]
        elif probe["probe_kind"] == "absent":
            assert graded["correct"]


def test_inventing_an_answer_to_an_absent_question_scores_wrong():
    probe = next(p for p in probes(build_script(SPEC, 100, 0)) if p["probe_id"] == "exam-date")

    assert scoring.score_probe(probe, "Your exam is on 4 January.", REFUSAL) == {"correct": False, "reason": "invented an answer"}


def test_the_old_value_alone_is_a_stale_answer():
    probe = next(p for p in probes(build_script(SPEC, 100, 0)) if p["probe_id"] == "slot")

    assert scoring.score_probe(probe, "Your slot is on 3 December.", REFUSAL)["reason"] == "stale value"
    assert scoring.score_probe(probe, "It moved from 3 December to 10 December.", REFUSAL)["correct"]
    assert scoring.score_probe(probe, "Your slot is on 13 December.", REFUSAL)["reason"] != "stale value"


def test_a_trap_answer_that_endorses_the_false_premise_is_wrong():
    probe = next(p for p in probes(build_script(SPEC, 100, 0)) if p["probe_id"] == "romney-lcc")

    endorsed = "Romney Marsh v. Trinity House held that the last clear chance doctrine applies to the shipowner."

    assert scoring.score_probe(probe, endorsed, REFUSAL)["reason"] == "endorsed the false premise"
    assert scoring.score_probe(probe, "Plain nonsense about the weather.", REFUSAL)["reason"] == "no correction"


def test_empty_answers_and_unknown_kinds():
    assert scoring.score_probe({"probe_kind": "recall"}, "  ", REFUSAL)["reason"] == "empty answer"

    with pytest.raises(ValueError):
        scoring.score_probe({"probe_kind": "dance"}, "x", REFUSAL)


def test_page_hit():
    assert scoring.page_hit(None, [{"page": 5}]) is None
    assert scoring.page_hit([96, 99], [{"page": 5}, {"page": 98}]) is True
    assert scoring.page_hit([96, 99], [{"page": 100}, {"page": "N/A"}]) is False
    assert scoring.page_hit([96, 99], None) is False


# ---------- variants ----------

def test_groups_resolve_to_known_variants_without_duplicates():
    for group in variants.GROUPS:
        assert set(variants.resolve(group)) <= set(variants.VARIANTS)

    assert variants.resolve("full,main,full") == ["full", "no_memory", "full_history", "summary_buffer", "archive"]

    with pytest.raises(ValueError):
        variants.resolve("nonsense")


def test_variant_settings_name_real_settings():
    from app.core.config import settings

    for variant in variants.VARIANTS.values():
        for name in variant.settings:
            assert hasattr(settings, name), (variant.name, name)


def test_baselines_use_no_snapshots_and_snaprag_variants_do():
    assert not any(variants.VARIANTS[n].uses_snapshots for n in variants.GROUPS["baselines"])
    assert all(variants.VARIANTS[n].uses_snapshots for n in variants.GROUPS["ladder"])


def test_the_reconstruction_ladder_changes_only_snapshot_recall():
    ladder = [variants.VARIANTS[n].settings for n in ("snapshot", "evidence", "verify", "archive")]

    assert all(s["MEMORY_STATE_ENABLED"] is False for s in ladder)
    assert [s["MEMORY_RECALL_MODE"] for s in ladder] == ["snapshot", "evidence", "verify", "archive"]


# ---------- report ----------

def probe_row(id, kind, correct, bucket=None, excluded=False, reason="x"):
    return {"id": id, "kind": kind, "bucket": bucket, "correct": correct, "excluded": excluded, "reason": reason}


def make_run(variant, probes_, repeat=0, turns=()):
    return {"variant": variant, "repeat": repeat, "probes": probes_, "turns": list(turns), "compression": []}


def test_bootstrap_interval_brackets_the_mean_and_handles_edges():
    low, high = report.bootstrap_ci([1.0] * 30 + [0.0] * 30)

    assert low < 0.5 < high and 0.35 < low and high < 0.65
    assert report.bootstrap_ci([]) is None
    assert report.bootstrap_ci([1.0, 1.0]) == (1.0, 1.0)


def test_summary_counts_buckets_kinds_and_leaves_out_excluded_probes():
    rows = [probe_row("a", "recall", True, "near"), probe_row("b", "recall", False, "very_far", reason="forgot"),
            probe_row("c", "absent", True), probe_row("d", "recall", False, "near", excluded=True)]

    summary = report.summarize_variant([make_run("v", rows)])

    assert summary["overall"]["n"] == 3 and summary["overall"]["accuracy"] == pytest.approx(2 / 3)
    assert summary["by_bucket"]["near"]["accuracy"] == 1.0 and summary["by_bucket"]["very_far"]["accuracy"] == 0.0
    assert summary["memory_loss"] == 1.0 and summary["honesty"]["accuracy"] == 1.0
    assert summary["failures"] == {"forgot": 1} and summary["excluded_probes"] == 1


def test_summary_averages_cost_per_turn_and_skips_failed_turns():
    turns = [{"seconds": 2.0, "query_calls": 4, "maintenance_calls": 1, "prompt_tokens": 1000, "page_hit": True},
             {"seconds": 4.0, "query_calls": 6, "maintenance_calls": 3, "prompt_tokens": 3000, "page_hit": False},
             {"seconds": 99.0, "query_calls": 0, "maintenance_calls": 0, "prompt_tokens": None, "error": "boom"}]

    s = report.summarize_variant([make_run("v", [], turns=turns)])

    assert (s["seconds"], s["query_calls"], s["maintenance_calls"], s["prompt_tokens"]) == (3.0, 5.0, 2.0, 2000.0)
    assert s["doc_page_hit"] == 0.5 and s["errors"] == 1


def test_markdown_table_has_a_row_per_variant():
    summary = report.summarize([make_run("alpha", [probe_row("a", "recall", True, "near")]),
                                make_run("beta", [probe_row("a", "recall", False, "near", reason="forgot")])])

    text = report.render_markdown(summary)

    assert "| alpha | 100%" in text and "| beta | 0%" in text and "beta: forgot 1" in text


def test_paired_difference_is_computed_over_shared_probes():
    a = make_run("a", [probe_row("x", "recall", True), probe_row("y", "recall", True), probe_row("z", "recall", False)])
    b = make_run("b", [probe_row("x", "recall", False), probe_row("y", "recall", True), probe_row("z", "recall", False)])

    diff = report.paired_difference([a], [b])

    assert diff["n"] == 3 and diff["difference"] == pytest.approx(1 / 3)
    assert report.paired_difference([a], [make_run("c", [probe_row("other", "recall", True)])]) is None


# ---------- runner ----------

class StubAsk:
    """A system that answers every probe from the gold answer, except turns in `fail`."""

    def __init__(self, script, fail=()):
        self.script = {t["text"]: t for t in script}
        self.fail = set(fail)
        self.histories = []

    def __call__(self, question, history, session):
        from app.core.exceptions import LLMProviderError

        self.histories.append(history)
        item = self.script[question]

        if item["turn"] in self.fail:
            raise LLMProviderError("rate limited")

        answer = item["gold"] if item["kind"] == "probe" else "ok"

        return {"answer": answer, "citations": [{"page": 100}], "verification": {"supported": True}, "confidence": 0.9}


def play_stub(script, ask, variant="no_memory", **kwargs):
    from app.services.session_store import create_chat

    conversation = run.Conversation(variants.VARIANTS[variant], "bench", create_chat("bench"), upkeep=False)

    return run.play(variants.VARIANTS[variant], script, SPEC, ask=ask, conversation=conversation,
                    document="d.pdf", retries=2, wait=0, log=lambda *a: None, **kwargs), conversation


def test_play_scores_every_probe_and_keeps_chat_turns_equal_to_script_turns():
    script = build_script(SPEC, 80, 0)

    result, conversation = play_stub(script, StubAsk(script))

    assert len(result["probes"]) == 24 and all(p["correct"] for p in result["probes"])
    assert [t["turn"] for t in conversation.turns()] == list(range(1, 81))


def test_a_failed_turn_is_excluded_not_counted_against_the_system():
    script = build_script(SPEC, 80, 0)
    source = next(p for p in probes(script) if p["probe_id"] == "deadline")["source_turns"][0]

    result, conversation = play_stub(script, StubAsk(script, fail={source}))

    deadline = next(p for p in result["probes"] if p["id"] == "deadline")

    assert deadline["excluded"] is True
    assert sum(1 for p in result["probes"] if p["excluded"]) == 1
    assert [t["turn"] for t in conversation.turns()] == list(range(1, 81))          # numbering survives
    assert result["turns"][source - 1]["error"] == "rate limited"


def test_a_probe_whose_own_answer_failed_is_excluded_and_wrong():
    script = build_script(SPEC, 80, 0)
    probe = next(p for p in probes(script) if p["probe_id"] == "deadline")

    result, _ = play_stub(script, StubAsk(script, fail={probe["turn"]}))

    row = next(p for p in result["probes"] if p["id"] == "deadline")

    assert row["excluded"] and not row["correct"] and row["reason"] == "service unavailable"


def test_the_dry_run_system_reproduces_its_memory_curve():
    script = build_script(SPEC, 100, 0)
    oracle = DryRunSystem(script, remembers=20)
    clock = {"turn": 0}

    def ask(question, history, session):
        clock["turn"] += 1
        return oracle.answer(question, clock["turn"])

    result, _ = play_stub(script, ask)
    summary = report.summarize_variant([result])

    assert summary["by_bucket"]["near"]["accuracy"] == 1.0
    assert summary["by_bucket"]["very_far"]["accuracy"] == 0.0
    assert summary["honesty"]["accuracy"] == 1.0


def test_summary_buffer_folds_turns_that_leave_the_window():
    folded = []
    buffer = run.SummaryBuffer(lambda text, turn: folded.append(turn["turn"]) or f"{text}+{turn['turn']}", keep=2)
    turns = [{"turn": i, "question": f"q{i}", "answer": f"a{i}"} for i in range(1, 6)]

    buffer.update(turns)

    assert folded == [1, 2, 3] and buffer.text == "+1+2+3"
    history = buffer.history(turns)
    assert history[0]["answer"] == "+1+2+3" and [h["turn"] for h in history[1:]] == [4, 5]

    buffer.update(turns)                                                   # nothing new to fold
    assert folded == [1, 2, 3]


def test_settings_overrides_are_restored():
    from app.core.config import settings

    before = settings.HISTORY_TURNS

    with run.overridden({"HISTORY_TURNS": 99}):
        assert settings.HISTORY_TURNS == 99

    assert settings.HISTORY_TURNS == before


def test_comparisons_name_each_variant_against_the_reference():
    runs = [make_run("no_memory", [probe_row("x", "recall", False)]), make_run("full", [probe_row("x", "recall", True)])]

    lines = run.comparisons(runs)

    assert any(line.startswith("- full: +100.0%") for line in lines)
    assert run.comparisons(runs[1:]) == []


def test_save_and_resume_roundtrip(tmp_path):
    import json

    path = str(tmp_path / "r" / "out.json")
    runs = [make_run("full", [probe_row("x", "recall", True)])]

    run.save(path, runs, {"turns": 100})

    assert json.load(open(path, encoding="utf-8"))["runs"] == runs
