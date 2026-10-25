"""Builds the figures and the example boxes of the paper from the result files.

    cd rag-service && ..\\venv\\Scripts\\python.exe ..\\paper\\make_figures.py

Figures: architecture.pdf (always), distance.pdf (accuracy by distance bucket), ladder.pdf (the verification ladder),
docqa.pdf (accuracy and cost of the architectures). The example boxes (a failure the memory fixes, a sentence kept and one
dropped) are written to generated/ as LaTeX from real rows. A figure whose data does not exist yet is skipped.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RAG = os.path.join(os.path.dirname(HERE), "rag-service")
RESULTS = os.path.join(RAG, "benchmarks", "results")
FIG = os.path.join(HERE, "figures")
GEN = os.path.join(HERE, "generated")
sys.path.insert(0, RAG)

from benchmarks.docqa import stats as S

BUCKETS = ("near", "mid", "far", "very_far")
MEMORY_KINDS = ("recall", "supersede", "reference")
COLORS = ["#1f3a5f", "#c0504d", "#4f9a5c", "#e0a030", "#7a5195", "#2a9d8f", "#999999", "#8c564b"]


def load(name):

    path = os.path.join(RESULTS, name)

    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def box(ax, x, y, w, h, text, color="#e8eef6"):

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#1f3a5f", lw=1))
    ax.text(x + w / 2, y + h / 2, text.replace("|", "\n"), ha="center", va="center", fontsize=7)


def arrow(ax, x0, y0, x1, y1):

    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=1))


def architecture():

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box(ax, 0.2, 4.2, 1.6, 1.0, "question")
    box(ax, 2.3, 4.2, 1.9, 1.0, "intent and plan|(first batch, budget)")
    box(ax, 4.9, 5.2, 2.4, 1.0, "document retrieval|dense + BM25 + rerank")
    box(ax, 4.9, 3.2, 2.4, 1.2, "memory recall: verified|state + verified|reconstruction", "#f3e6e6")
    box(ax, 8.0, 4.2, 1.6, 1.0, "answer from|fused context")
    box(ax, 10.1, 4.2, 1.7, 1.0, "confidence check|(AB-RAG)")
    box(ax, 8.0, 6.0, 3.8, 0.8, "below threshold and budget left:|read the next batch of pages", "#fdf3dc")
    box(ax, 0.2, 0.3, 3.2, 1.3, "cold archive|(every turn, immutable)", "#e3f1e6")
    box(ax, 3.9, 0.3, 3.4, 1.3, "snapshots (lossy, hash-chained,|anchored to document pages)", "#e3f1e6")
    box(ax, 7.8, 0.3, 4.0, 1.3, "after each turn: the context-quality score|decides whether to snapshot;|user statements go to the verified state", "#e3f1e6")
    for a in ((1.8, 4.7, 2.3, 4.7), (4.2, 4.9, 4.9, 5.6), (4.2, 4.5, 4.9, 3.9), (7.3, 5.6, 8.0, 4.9), (7.3, 3.9, 8.0, 4.4), (9.6, 4.7, 10.1, 4.7),
              (10.95, 5.2, 10.95, 6.0), (8.0, 6.4, 7.3, 6.0), (3.4, 0.95, 3.9, 0.95), (5.6, 1.6, 5.9, 3.2)):
        arrow(ax, *a)
    ax.text(6.0, 6.85, "SnapRAG: one turn", ha="center", fontsize=10, weight="bold")
    fig.savefig(os.path.join(FIG, "architecture.pdf"), bbox_inches="tight")
    plt.close(fig)


def distance():

    names = [n for n in ("part1", "part2") if load(f"longconv_{n}.json")]

    if not names:
        return

    runs = [r for n in names for r in load(f"longconv_{n}.json")["runs"]]
    variants = list(dict.fromkeys(r["variant"] for r in runs))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.8 / max(1, len(variants))

    for i, variant in enumerate(variants):
        probes = [p for r in runs if r["variant"] == variant for p in r["probes"] if p["kind"] in MEMORY_KINDS and not p.get("excluded")]
        xs, ys, lows, highs = [], [], [], []
        for j, bucket in enumerate(BUCKETS):
            part = [p for p in probes if p.get("bucket") == bucket]
            if part:
                k, n = sum(p["correct"] for p in part), len(part)
                low, high = S.exact_interval(k, n)
                xs.append(j + i * width - 0.4 + width / 2)
                ys.append(100 * k / n)
                lows.append(100 * k / n - 100 * low)
                highs.append(100 * high - 100 * k / n)
        ax.bar(xs, ys, width * 0.95, yerr=[lows, highs], label=variant, color=COLORS[i % len(COLORS)], capsize=2, error_kw={"lw": 0.8})

    ax.set_xticks(range(4))
    ax.set_xticklabels(["near (3-10)", "mid (12-25)", "far (30-50)", "very far (60-85)"])
    ax.set_ylabel("memory-probe accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=3, loc="lower left")
    fig.savefig(os.path.join(FIG, "distance.pdf"), bbox_inches="tight")
    plt.close(fig)


def ladder():

    data = load("ladder_full.json")

    if not data:
        return

    modes = list(data["summary"])
    rates, recalls = [], []

    for mode in modes:
        rows = [r for r in data["rows"] if r.get("mode") == mode and r.get("unsupported") is not None]
        returned = sum(r["returned"] for r in rows)
        rates.append(100 * sum(r["unsupported"] for r in rows) / returned if returned else 0)
        values = [r["fact_recall"] for r in data["rows"] if r.get("mode") == mode and r.get("fact_recall") is not None]
        recalls.append(100 * sum(values) / len(values) if values else 0)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
    axes[0].bar(modes, rates, color="#c0504d")
    axes[0].set_ylabel("unsupported sentences (%)")
    axes[1].bar(modes, recalls, color="#4f9a5c")
    axes[1].set_ylabel("fact recall (%)")

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ladder.pdf"), bbox_inches="tight")
    plt.close(fig)


def docqa():

    data = load("docqa_full75.json")

    if not data:
        return

    rows = [r for r in data["rows"] if "error" not in r and "correct" in r]
    archs = list(dict.fromkeys(r["arch"] for r in rows))
    archs = [a for a in archs if a not in ("snaprag_fixed_k_noplan", "snaprag_calib")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
    points = []

    for i, arch in enumerate(archs):
        mine = [r for r in rows if r["arch"] == arch]
        k, n = sum(r["correct"] for r in mine), len(mine)
        low, high = S.exact_interval(k, n)
        axes[0].barh(i, 100 * k / n, xerr=[[100 * k / n - 100 * low], [100 * high - 100 * k / n]], color="#1f3a5f" if arch.startswith("snaprag") else "#9db4d0", capsize=2, error_kw={"lw": 0.8})
        calls = [r["calls"] for r in mine if "calls" in r]
        points.append((arch, sum(calls) / len(calls) if calls else 0, 100 * k / n))

    axes[0].set_yticks(range(len(archs)))
    axes[0].set_yticklabels(archs, fontsize=7)
    axes[0].set_xlabel("accuracy (%), exact 95% interval")
    axes[0].invert_yaxis()

    for arch, calls, acc in points:
        axes[1].scatter(calls, acc, color="#c0504d" if arch.startswith("snaprag") else "#1f3a5f", s=18)
        axes[1].annotate(arch, (calls, acc), fontsize=5, xytext=(2, 2), textcoords="offset points")

    axes[1].set_xlabel("LLM calls per question")
    axes[1].set_ylabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "docqa.pdf"), bbox_inches="tight")
    plt.close(fig)


def tex(text: str) -> str:

    for char in "&%_#$":
        text = text.replace(char, "\\" + char)

    return " ".join(text.split())


def examples():
    """A failure the memory fixes, and a sentence the verifier kept and one it dropped, from real rows."""

    os.makedirs(GEN, exist_ok=True)
    runs = [r for n in ("part1", "part2") if load(f"longconv_{n}.json") for r in load(f"longconv_{n}.json")["runs"]]
    by = {r["variant"]: {p["id"]: p for p in r["probes"]} for r in runs}

    if "no_memory" in by and "full" in by:
        for pid, probe in by["full"].items():
            other = by["no_memory"].get(pid)
            if probe["correct"] and other and not other["correct"] and probe["kind"] in MEMORY_KINDS and other.get("answer") != probe.get("answer"):
                with open(os.path.join(GEN, "example_failure.tex"), "w", encoding="utf-8") as f:
                    f.write("\\begin{figure}[t]\\small\\fbox{\\parbox{0.95\\linewidth}{"
                            f"\\textbf{{Probe ({probe['bucket']}, turn {probe['turn']}):}} kind {tex(probe['kind'])}.\\\\"
                            f"\\textbf{{Last 3 turns only:}} {tex(other['answer'][:220])}\\\\"
                            f"\\textbf{{\\sys{{}}:}} {tex(probe['answer'][:220])}}}}}\n"
                            "\\caption{A memory probe the baseline forgets and \\sys{} answers (taken from the run).}\\label{fig:failure}\\end{figure}\n")
                break

    ladder_data = load("ladder_full.json")

    if ladder_data:
        kept = next((r for r in ladder_data["rows"] if r.get("mode") == "archive" and r.get("kept")), None)
        dropped = next((r for r in ladder_data["rows"] if r.get("mode") == "archive" and r.get("dropped_sentences")), None)
        if kept and dropped:
            gone = dropped["dropped_sentences"][0]
            gone = gone if isinstance(gone, str) else gone.get("text", "")
            with open(os.path.join(GEN, "example_verify.tex"), "w", encoding="utf-8") as f:
                f.write("\\begin{figure}[t]\\small\\fbox{\\parbox{0.95\\linewidth}{"
                        f"\\textbf{{Kept:}} {tex(kept['kept'][0]['text'])}\\\\\\textbf{{Dropped:}} {tex(gone)}}}}}\n"
                        "\\caption{One reconstructed sentence the verifier kept and one it dropped (from the ladder run).}\\label{fig:verify}\\end{figure}\n")


def main() -> None:

    os.makedirs(FIG, exist_ok=True)
    os.makedirs(GEN, exist_ok=True)

    for step in (architecture, distance, ladder, docqa, examples):
        try:
            step()
            print("ok", step.__name__)
        except Exception as exc:       # a figure must not stop the others
            print("skipped", step.__name__, type(exc).__name__, exc)


if __name__ == "__main__":
    main()
