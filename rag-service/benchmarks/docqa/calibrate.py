"""Calibrates SnapRAG's AB-RAG confidence on held-out questions.

The `snaprag_calib` architecture never stops early, so each question has every attempt (the first batch
and the deeper ones) with its three confidence signals and its answer. That lets any weights and threshold
be replayed offline without new LLM calls: the policy reads attempts in order, stops at the first whose
confidence reaches the threshold, and returns the most confident attempt it has read (as run_batches does).

    python -m benchmarks.docqa.calibrate benchmarks/results/docqa_full75.json --test-from benchmarks/results/docqa_pilot30.json

Weights and threshold are chosen on the questions NOT in the test sample and then scored on the test sample.
"""

import argparse
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from benchmarks.docqa import stats as S
from benchmarks.docqa.scoring import score_answer

DEFAULT = {"weights": (0.50, 0.25, 0.25), "threshold": 0.80}
THRESHOLDS = [round(0.50 + 0.025 * i, 3) for i in range(21)]                  # 0.50 .. 1.00
WEIGHTS = [(a / 4, b / 4, (4 - a - b) / 4) for a in range(5) for b in range(5 - a)]   # steps of 0.25, sum 1
EXTRA_ATTEMPT_PENALTY = 0.01        # accuracy points one more attempt must buy to be worth its LLM calls


def confidence(signals: dict, supported: bool, weights, threshold: float) -> float:
    """The AB-RAG confidence of an attempt under `weights`: the weighted mean of its signals,
    capped just below the threshold when the checker did not support the answer."""

    model, similarity, spread = (signals.get(k, 0.0) for k in ("model", "similarity", "spread"))
    total = sum(weights) or 1.0
    value = (weights[0] * model + weights[1] * similarity + weights[2] * spread) / total

    return round(min(value, threshold - 0.01) if not supported else value, 2)


def questions(rows: list[dict], dataset: dict) -> dict:
    """{question id: [attempt, ...]} with each attempt's correctness, from the snaprag_calib rows."""

    items = {q["id"]: q for q in dataset["questions"]}
    out = {}

    for row in rows:
        if row["arch"] == "snaprag_calib" and "error" not in row and row.get("attempts"):
            out[row["id"]] = [{**a, "correct": float(score_answer(items[row["id"]], a["answer"])["correct"])} for a in row["attempts"]]

    return out


def replay(attempts: list[dict], weights, threshold: float, limit: int | None = None) -> dict:
    """The outcome of the policy on one question's attempts (`limit` caps how many it may read)."""

    best, best_conf, read = None, -1.0, 0

    for attempt in attempts[:limit]:
        read += 1
        conf = confidence(attempt["signals"], attempt["supported"], weights, threshold)

        if conf > best_conf:
            best, best_conf = attempt, conf

        if conf >= threshold:
            break

    return {"correct": best["correct"], "attempts": read, "passages": attempts[read - 1]["passages"]}


def evaluate(data: dict, ids, weights, threshold, limit: int | None = None) -> dict:

    runs = [replay(data[i], weights, threshold, limit) for i in ids if i in data]

    return {"accuracy": sum(r["correct"] for r in runs) / len(runs), "attempts": sum(r["attempts"] for r in runs) / len(runs),
            "passages": sum(r["passages"] for r in runs) / len(runs), "n": len(runs), "right": sum(r["correct"] for r in runs)}


def search(data: dict, ids) -> list[dict]:
    """Every (weights, threshold) on the grid, scored on `ids`, best first. The score is accuracy minus a small
    price per extra attempt, so equal accuracy is won by the cheaper policy."""

    scored = []

    for weights, threshold in itertools.product(WEIGHTS, THRESHOLDS):
        result = evaluate(data, ids, weights, threshold)
        scored.append({"weights": weights, "threshold": threshold, **result,
                       "score": result["accuracy"] - EXTRA_ATTEMPT_PENALTY * (result["attempts"] - 1)})

    return sorted(scored, key=lambda r: (-r["score"], r["attempts"], r["threshold"]))


def signal_auroc(data: dict, ids) -> dict:
    """How well each signal of the FIRST attempt tells a right answer from a wrong one (0.5 = nothing)."""

    first = [data[i][0] for i in ids if i in data]
    labels = [int(a["correct"]) for a in first]

    return {name: S.auroc([a["signals"].get(name, 0.0) for a in first], labels) for name in ("model", "similarity", "spread")} | {
        "supported flag": S.auroc([float(a["supported"]) for a in first], labels)}


def render(rows: list[dict], dataset: dict, test_ids: set[str]) -> tuple[str, dict]:
    """(Markdown section, the chosen configuration)."""

    data = questions(rows, dataset)
    calibration_ids = [i for i in data if i not in test_ids]
    test = [i for i in data if i in test_ids]
    ranked = search(data, calibration_ids)
    chosen = ranked[0]
    config = {"weights": chosen["weights"], "threshold": chosen["threshold"]}

    def row(label, weights, threshold, ids, limit=None):
        r = evaluate(data, ids, weights, threshold, limit)
        low, high = S.exact_interval(r["right"], r["n"])
        return [label, f"{weights[0]:.2f} / {weights[1]:.2f} / {weights[2]:.2f}", f"{threshold:.2f}", f"{r['right']}/{r['n']} = {100 * r['accuracy']:.1f}% ({100 * low:.0f}-{100 * high:.0f})",
                f"{r['attempts']:.2f}", f"{r['passages']:.1f}"]

    def table(ids):
        return [row("default (uncalibrated)", DEFAULT["weights"], DEFAULT["threshold"], ids),
                row("calibrated", chosen["weights"], chosen["threshold"], ids),
                row("never read more (fixed-k)", DEFAULT["weights"], DEFAULT["threshold"], ids, limit=1),
                row("always read everything", DEFAULT["weights"], 2.0, ids)]

    def md(header, rows_):
        lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
        return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows_])

    header = ["Policy", "Weights (model / similarity / spread)", "Threshold", "Accuracy (exact 95% CI)", "Attempts per question", "Pages of the last attempt"]
    signal = signal_auroc(data, [i for i in data])
    first_conf = [confidence(data[i][0]["signals"], data[i][0]["supported"], DEFAULT["weights"], DEFAULT["threshold"]) for i in data]

    out = ["# Calibration of the AB-RAG confidence", "",
           f"Attempts were recorded for {len(data)} questions with `snaprag_calib` (it never stops early). The weights and the threshold were chosen on the "
           f"{len(calibration_ids)} questions that are not in the test sample and scored on the {len(test)} test questions, so the test accuracy is not tuned on. "
           f"The grid has {len(WEIGHTS)} weight triples (steps of 0.25) and {len(THRESHOLDS)} thresholds; the choice maximises accuracy minus {EXTRA_ATTEMPT_PENALTY} per extra attempt.", "",
           "## Do the signals carry information?", "",
           f"With the default weights, the first attempt's confidence ranges from {min(first_conf):.2f} to {max(first_conf):.2f}; "
           f"{sum(c >= DEFAULT['threshold'] for c in first_conf)} of {len(first_conf)} first attempts reach the default threshold {DEFAULT['threshold']}, so "
           "the check " + ("almost never asks for another batch." if sum(c >= DEFAULT["threshold"] for c in first_conf) > 0.9 * len(first_conf) else "does trigger.")
           + " AUROC of each signal of the first attempt for telling a right answer from a wrong one (0.5 = no information):", "",
           md(["Signal", "AUROC"], [[k, "-" if v is None else f"{v:.2f}"] for k, v in signal.items()]), "",
           "## Held-out calibration set", "", md(header, table(calibration_ids)), "",
           "## Test sample (not used for the choice)", "", md(header, table(test)), "",
           "## Best ten on the calibration set", "",
           md(["Weights", "Threshold", "Accuracy", "Attempts", "Score"],
              [[" / ".join(f"{w:.2f}" for w in r["weights"]), f"{r['threshold']:.2f}", f"{100 * r['accuracy']:.1f}%", f"{r['attempts']:.2f}", f"{r['score']:.3f}"] for r in ranked[:10]]), ""]

    return "\n".join(out), config


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="results file with snaprag_calib rows (all questions)")
    parser.add_argument("--test-from", required=True, help="results file whose question ids are the test sample")
    parser.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results", "docqa_calibration.md"))
    args = parser.parse_args()

    rows = json.load(open(args.results, encoding="utf-8"))["rows"]
    test_ids = set(json.load(open(args.test_from, encoding="utf-8"))["meta"]["ids"])
    dataset = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json"), encoding="utf-8"))
    text, config = render(rows, dataset, test_ids)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    print(args.out, config)


if __name__ == "__main__":
    main()
