"""Aggregates benchmark runs into one table per variant.

A run is {"variant", "repeat", "probes": [...], "turns": [...]}; see run.py.
Repeats of a variant are pooled, and every accuracy comes with a bootstrap
95% confidence interval over its probes, so a difference between two
variants can be judged against the noise instead of by eye.
"""

import random
import statistics

KINDS = ("recall", "supersede", "reference", "absent", "trap")
BUCKETS = ("near", "mid", "far", "very_far")
MEMORY_KINDS = ("recall", "supersede", "reference")       # probes whose answer sits in earlier turns


def bootstrap_ci(values: list[float], iterations: int = 2000, seed: int = 0, alpha: float = 0.05):
    """(low, high) percentile interval of the mean, or None for no values."""

    if not values:
        return None

    if len(set(values)) == 1:
        return values[0], values[0]

    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(iterations))

    return means[int(alpha / 2 * iterations)], means[int((1 - alpha / 2) * iterations) - 1]


def _mean(values: list[float]):

    return sum(values) / len(values) if values else None


def _accuracy(rows: list[dict]) -> dict:

    scored = [1.0 if r["correct"] else 0.0 for r in rows if not r.get("excluded")]
    ci = bootstrap_ci(scored)

    return {"accuracy": _mean(scored), "n": len(scored), "ci": ci}


def _percentile(values: list[float], q: float):

    if not values:
        return None

    ordered = sorted(values)

    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize_variant(runs: list[dict]) -> dict:
    """Pools the repeats of one variant."""

    probes = [p for run in runs for p in run["probes"]]
    turns = [t for run in runs for t in run["turns"]]
    scored_turns = [t for t in turns if not t.get("error")]

    memory_probes = [p for p in probes if p["kind"] in MEMORY_KINDS]
    near, very_far = (
        _accuracy([p for p in memory_probes if p.get("bucket") == bucket])["accuracy"] for bucket in ("near", "very_far")
    )

    page_hits = [1.0 if t["page_hit"] else 0.0 for t in scored_turns if t.get("page_hit") is not None]
    seconds = [t["seconds"] for t in scored_turns if t.get("seconds") is not None]

    return {
        "repeats": len(runs),
        "overall": _accuracy(probes),
        "memory": _accuracy(memory_probes),
        "honesty": _accuracy([p for p in probes if p["kind"] in ("absent", "trap")]),
        "by_kind": {kind: _accuracy([p for p in probes if p["kind"] == kind]) for kind in KINDS},
        "by_bucket": {bucket: _accuracy([p for p in memory_probes if p.get("bucket") == bucket]) for bucket in BUCKETS},
        # How much worse the farthest facts are remembered than the nearest.
        "memory_loss": None if near is None or very_far is None else near - very_far,
        "doc_page_hit": _mean(page_hits),
        "prompt_tokens": _mean([t["prompt_tokens"] for t in scored_turns if t.get("prompt_tokens") is not None]),
        "query_calls": _mean([t["query_calls"] for t in scored_turns if t.get("query_calls") is not None]),
        "maintenance_calls": _mean([t["maintenance_calls"] for t in scored_turns if t.get("maintenance_calls") is not None]),
        "seconds": _mean(seconds),
        "seconds_p95": _percentile(seconds, 0.95),
        "errors": sum(1 for t in turns if t.get("error")),
        "excluded_probes": sum(1 for p in probes if p.get("excluded")),
        "compression": _mean([c for run in runs for c in run.get("compression", []) if c]),
        "failures": _failure_reasons(probes),
    }


def _failure_reasons(probes: list[dict]) -> dict:

    reasons = {}

    for probe in probes:
        if not probe["correct"] and not probe.get("excluded"):
            key = probe["reason"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1

    return reasons


def summarize(runs: list[dict]) -> dict:
    """{variant: summary}, in the order the variants first appear."""

    by_variant = {}

    for run in runs:
        by_variant.setdefault(run["variant"], []).append(run)

    return {variant: summarize_variant(group) for variant, group in by_variant.items()}


# ==========================================================
# Text
# ==========================================================

def _pct(cell: dict | None) -> str:

    if not cell or cell["accuracy"] is None:
        return "-"

    text = f"{cell['accuracy']:.0%}"

    if cell["ci"] and cell["ci"][0] != cell["ci"][1]:
        text += f" ({cell['ci'][0]:.0%}-{cell['ci'][1]:.0%})"

    return text


def _num(value, fmt="{:.1f}") -> str:

    return "-" if value is None else fmt.format(value)


def render_markdown(summary: dict) -> str:
    """Two tables: accuracy, then cost."""

    lines = [
        "### Probe accuracy (95% bootstrap interval)",
        "",
        "| variant | overall | memory probes | recall | supersede | reference | absent | trap | near | mid | far | very far | memory loss |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for variant, s in summary.items():
        loss = "-" if s["memory_loss"] is None else f"{s['memory_loss']:+.0%}"
        lines.append(
            f"| {variant} | {_pct(s['overall'])} | {_pct(s['memory'])} | "
            + " | ".join(_pct(s["by_kind"][kind]) for kind in KINDS) + " | "
            + " | ".join(_pct(s["by_bucket"][bucket]) for bucket in BUCKETS)
            + f" | {loss} |"
        )

    lines += [
        "",
        "### Cost and quality of the document answers",
        "",
        "| variant | prompt tokens/turn | LLM calls/turn (answer) | LLM calls/turn (memory upkeep) | seconds/turn | p95 s | doc page hit | compression | errors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for variant, s in summary.items():
        hit = "-" if s["doc_page_hit"] is None else f"{s['doc_page_hit']:.0%}"
        comp = "-" if s["compression"] is None else f"{s['compression']:.1f}x"
        lines.append(
            f"| {variant} | {_num(s['prompt_tokens'], '{:.0f}')} | {_num(s['query_calls'])} | {_num(s['maintenance_calls'])} | "
            f"{_num(s['seconds'])} | {_num(s['seconds_p95'])} | {hit} | {comp} | {s['errors']} |"
        )

    lines += ["", "### Why probes failed", ""]

    for variant, s in summary.items():
        lines.append(f"- {variant}: " + (", ".join(f"{k} {v}" for k, v in sorted(s["failures"].items())) or "none"))

    return "\n".join(lines) + "\n"


def paired_difference(runs_a: list[dict], runs_b: list[dict], iterations: int = 2000, seed: int = 0):
    """Accuracy of A minus accuracy of B over the probes they share (same
    script and seed), with a bootstrap interval. Paired, so it is much
    tighter than comparing two separate intervals. None without shared probes."""

    def by_probe(runs):
        table = {}
        for run in runs:
            for p in run["probes"]:
                if not p.get("excluded"):
                    table.setdefault((run["repeat"], p["id"]), []).append(1.0 if p["correct"] else 0.0)
        return {key: statistics.mean(values) for key, values in table.items()}

    a, b = by_probe(runs_a), by_probe(runs_b)
    shared = sorted(set(a) & set(b))

    if not shared:
        return None

    diffs = [a[key] - b[key] for key in shared]

    return {"difference": sum(diffs) / len(diffs), "ci": bootstrap_ci(diffs, iterations, seed), "n": len(diffs)}
