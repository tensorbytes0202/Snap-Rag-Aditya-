"""The document-QA comparison: dataset shape, scoring and the report."""

import json
import os
import re

import pytest

from benchmarks.docqa import stats
from benchmarks.docqa.export import to_latex
from benchmarks.docqa.report import render
from benchmarks.docqa.scoring import score_answer, score_retrieval


DATASET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks", "docqa", "dataset.json")

ITEM = {"id": "q1", "type": "detail", "difficulty": "easy", "question": "?", "answer": "",
        "pages": [11, 31], "facts": [{"a": r"Prem\s+Nivas"}, {"a": r"Pasand\s+Bagh|Pasand"}]}
FALSE_PREMISE = {"id": "q2", "type": "false_premise", "difficulty": "hard", "question": "?", "answer": "", "pages": [33],
                 "facts": [], "answerable": False, "correction": r"\bdead\b|\bdied\b"}


# ---------- dataset ----------

def test_dataset_is_well_formed():
    data = json.load(open(DATASET, encoding="utf-8"))
    items = data["questions"]

    assert len({i["id"] for i in items}) == len(items) >= 50

    for item in items:
        assert item["type"] and item["difficulty"] in ("easy", "medium", "hard")
        assert item["question"].endswith("?") and item["answer"]

        if item.get("answerable", True):
            assert item["pages"] and item["facts"]
            for fact in item["facts"]:
                re.compile(fact["a"])
                re.compile(fact.get("e", fact["a"]))
        else:
            assert not item["facts"]

    assert sum(not i.get("answerable", True) for i in items) >= 5


def test_every_reference_answer_scores_as_correct():
    data = json.load(open(DATASET, encoding="utf-8"))

    assert [i["id"] for i in data["questions"] if not score_answer(i, i["answer"])["correct"]] == []


# ---------- scoring ----------

def test_an_answer_needs_every_fact():
    full = score_answer(ITEM, "It is held at Prem Nivas in Pasand Bagh.")
    half = score_answer(ITEM, "It is held at Prem Nivas.")

    assert full["correct"] and full["score"] == 1.0
    assert not half["correct"] and half["score"] == 0.5


def test_refusing_an_answerable_question_is_counted():
    scored = score_answer(ITEM, "The document does not mention this.")

    assert scored == {"correct": False, "score": 0.0, "declined": True}


def test_unanswerable_questions_are_handled_by_declining_or_correcting():
    assert score_answer(FALSE_PREMISE, "The text does not say that.")["correct"]
    assert score_answer(FALSE_PREMISE, "He had died eight years earlier.")["correct"]
    assert not score_answer(FALSE_PREMISE, "He gave her away with a handful of petals.")["correct"]


def test_retrieval_is_scored_on_the_context_and_the_pages_read():
    scored = score_retrieval(ITEM, "... at Prem Nivas, a house ...", [11, 99])

    assert scored == {"context_recall": 0.5, "page_hit": True, "page_recall": 0.5}
    assert score_retrieval(FALSE_PREMISE, "anything", [33]) is None


# ---------- report ----------

def row(arch, id, correct, recall=1.0):
    return {"arch": arch, "id": id, "type": "detail", "difficulty": "easy", "answerable": True, "answer": "a",
            "correct": correct, "score": float(correct), "declined": False, "context_recall": recall, "page_hit": True,
            "page_recall": 1.0, "passages": 5, "context_chars": 1000, "calls": 2, "seconds": 3.0, "pages": [1]}


def test_report_compares_architectures_on_the_same_questions():
    ids = [f"q{i}" for i in range(12)]
    rows = [row("dense_k5", i, n % 3 == 0, 0.5) for n, i in enumerate(ids)] + [row("snaprag", i, True, 1.0) for i in ids]
    result = {"meta": {"document": "d.pdf", "model": "m", "started": "now", "questions": 12,
                       "archs": {"dense_k5": "dense", "snaprag": "full"}}, "rows": rows}

    text = render(result)

    assert "## Answer quality" in text and "## Retrieval quality" in text
    assert "`snaprag` against every other architecture" in text
    assert "better" in text                      # the paired interval excludes zero
    assert "| `snaprag` | **12/12 = 100.0% (74-100)**" in text       # an exact interval, not (100-100)
    assert "Holm" in text


def test_a_retrieval_only_report_has_no_answer_tables():
    result = {"meta": {"document": "d.pdf", "model": "m", "started": "now", "questions": 2, "retrieval_only": True,
                       "archs": {"dense_k5": "dense"}},
              "rows": [{k: v for k, v in row("dense_k5", f"q{i}", True).items() if k not in ("correct", "score", "declined", "answer")}
                       for i in range(2)]}

    text = render(result)

    assert "## Retrieval quality" in text and "## Answer quality" not in text


# ---------- statistics ----------

def test_exact_intervals_do_not_collapse_at_zero_or_all():
    low, high = stats.exact_interval(12, 12)

    assert high == 1.0 and 0.7 < low < 0.8                    # 12 of 12 is not "100% (100-100)"
    assert stats.exact_interval(0, 30)[0] == 0.0 and 0.1 < stats.exact_interval(0, 30)[1] < 0.2


def test_sign_test_and_holm_adjustment():
    assert stats.sign_p(8, 0) == pytest.approx(2 / 256)
    assert stats.sign_p(0, 0) == 1.0
    assert stats.holm({"a": 0.01, "b": 0.04, "c": 0.5}) == pytest.approx({"a": 0.03, "b": 0.08, "c": 0.5})


def test_paired_p_uses_the_sign_test_for_outcomes_and_wilcoxon_for_scores():
    assert stats.paired_p([1, 1, 1, 1, 0, 0]) == stats.sign_p(4, 0)
    assert stats.paired_p([0.3, 0.2, 0.4, 0.1, 0.25, 0.35, 0.15, 0.2]) < 0.05
    assert stats.paired_p([0, 0, 0]) == 1.0


def test_auroc_of_a_confidence():
    assert stats.auroc([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0]) == 1.0
    assert stats.auroc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == 0.5
    assert stats.auroc([0.9, 0.8], [1, 1]) is None


BS = chr(92)


def test_latex_tables_scale_to_the_page_and_have_unique_labelled_captions():
    markdown = "\n".join([
        '# Run A', '', '## Answer quality', '', '| Architecture | Accuracy |', '|---|---|', '| `x_y` | 5/10 = 50.0% (24-76) |', '',
        '# Run B', '', '## Answer quality', '', '| Architecture | Accuracy |', '|---|---|', '| `x_y` | 1/2 |', ''])

    tex = to_latex(markdown)

    assert tex.count(BS + 'resizebox') == 2 and tex.count(BS + 'label{tab:') == 2
    assert 'x' + BS + '_y' in tex and 'Run A -- Answer quality' in tex and 'Run B -- Answer quality' in tex


# ---------- calibration ----------

def attempt(level, correct, passages):
    return {"signals": {"model": level, "similarity": level, "spread": level}, "supported": True, "passages": passages, "correct": float(correct)}


def test_calibration_replays_attempts_and_prefers_the_threshold_that_reads_more_when_it_helps():
    from benchmarks.docqa import calibrate

    data = {f"q{i}": [attempt(0.5, 0, 3), attempt(0.9, 1, 6)] for i in range(6)}

    assert calibrate.replay(data["q0"], (0.5, 0.25, 0.25), 0.8)["correct"] == 1.0         # 0.5 < 0.8, so it reads the second attempt
    assert calibrate.replay(data["q0"], (0.5, 0.25, 0.25), 0.4)["correct"] == 0.0         # 0.5 >= 0.4, so it stops at the first
    assert calibrate.replay(data["q0"], (0.5, 0.25, 0.25), 0.8, limit=1)["attempts"] == 1

    best = calibrate.search(data, list(data))[0]

    assert best["accuracy"] == 1.0 and best["threshold"] > 0.5


def test_an_unsupported_answer_never_reaches_the_threshold():
    from benchmarks.docqa import calibrate

    assert calibrate.confidence({"model": 1, "similarity": 1, "spread": 1}, False, (0.5, 0.25, 0.25), 0.8) == 0.79


def test_memqa_scoring():
    from benchmarks.memqa.run import f1, score

    assert f1("She went on 7 May 2023.", "7 May 2023") >= 0.5
    assert score({"abstain": False, "gold": "GPS system"}, "The gps system failed.")["correct"]
    assert score({"abstain": True, "gold": None}, "That is not mentioned in the conversation.")["correct"]
    assert not score({"abstain": True, "gold": None}, "She chose Paris.")["correct"]
