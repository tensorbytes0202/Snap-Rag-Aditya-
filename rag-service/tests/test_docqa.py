"""The document-QA comparison: dataset shape, scoring and the report."""

import json
import os
import re

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
    assert "| `snaprag` | **100.0% (100-100)**" in text


def test_a_retrieval_only_report_has_no_answer_tables():
    result = {"meta": {"document": "d.pdf", "model": "m", "started": "now", "questions": 2, "retrieval_only": True,
                       "archs": {"dense_k5": "dense"}},
              "rows": [{k: v for k, v in row("dense_k5", f"q{i}", True).items() if k not in ("correct", "score", "declined", "answer")}
                       for i in range(2)]}

    text = render(result)

    assert "## Retrieval quality" in text and "## Answer quality" not in text
