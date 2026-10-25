"""The long-conversation results as one report: exact intervals, paired tests with Holm adjustment, measured tokens.

    python -m benchmarks.longconv.final_report benchmarks/results/longconv_part1.json benchmarks/results/longconv_part2.json \
        --ladder benchmarks/results/ladder_full.json --audit audit_sheet.json --pdf out.pdf --latex out.tex

Runs from several result files are pooled by variant. Proportions come with exact (Clopper-Pearson) intervals; the
comparisons are paired on the same probes (same script and seed) with an exact sign test and Holm adjustment.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from benchmarks.docqa import stats as S
from benchmarks.docqa.export import to_latex, write_pdf
from benchmarks.longconv.report import BUCKETS, KINDS, MEMORY_KINDS

REFERENCES = ("no_memory", "full")
ALPHA = 0.05


def _mean(values):

    values = [v for v in values if v is not None]

    return sum(values) / len(values) if values else None


def _num(value, fmt="{:.1f}"):

    return "-" if value is None else fmt.format(value)


def _cell(rows) -> str:
    """'k/n = p% (exact 95% interval)' for a list of probes."""

    rows = [r for r in rows if not r.get("excluded")]

    if not rows:
        return "-"

    k, n = sum(1 for r in rows if r["correct"]), len(rows)
    low, high = S.exact_interval(k, n)

    return f"{k}/{n} = {100 * k / n:.0f}% ({100 * low:.0f}-{100 * high:.0f})"


def _table(header, rows):

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]

    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows])


def _percentile(values, q):

    values = sorted(values)

    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def pool(paths: list[str]) -> tuple[dict, dict]:
    """({variant: [runs]}, meta of the first file) from several result files."""

    by_variant, meta = {}, {}

    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        meta = meta or data.get("meta", {})
        for run in data["runs"]:
            by_variant.setdefault(run["variant"], []).append(run)

    return by_variant, meta


def probes_of(runs):
    return [p for run in runs for p in run["probes"]]


def paired(a_runs, b_runs):
    """Right/wrong of A against B over the probes they share: wins, losses, ties, difference and p."""

    def table(runs):
        return {(run["repeat"], p["id"]): float(p["correct"]) for run in runs for p in run["probes"] if not p.get("excluded")}

    a, b = table(a_runs), table(b_runs)
    shared = sorted(set(a) & set(b))

    if not shared:
        return None

    diffs = [a[k] - b[k] for k in shared]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)

    return {"mean": sum(diffs) / len(diffs), "ci": S.bootstrap_ci(diffs), "wins": wins, "losses": losses, "n": len(diffs), "p": S.sign_p(wins, losses)}


def cost_row(runs) -> list[str]:

    turns = [t for run in runs for t in run["turns"] if not t.get("error")]
    hits = [float(t["page_hit"]) for t in turns if t.get("page_hit") is not None]
    active = [t["active_seconds"] for t in turns if t.get("active_seconds") is not None]
    calls = [t["query_calls"] + t.get("maintenance_calls", 0) for t in turns if t.get("query_calls") is not None]
    compression = [c for run in runs for c in run.get("compression", []) if c]
    tokens_in = [t["tokens_in"] for t in turns if t.get("tokens_in")]
    tokens_out = [t["tokens_out"] for t in turns if t.get("tokens_out")]

    return [_num(_mean(tokens_in), "{:,.0f}"), _num(_mean(tokens_out), "{:,.0f}"), _num(_mean(calls)),
            "-" if not active else f"{_num(_mean(active))} / {_num(_percentile(active, 0.95))}",
            "-" if not hits else f"{100 * _mean(hits):.0f}%", _num(_mean([run.get("snapshots") for run in runs])),
            "-" if not compression else f"{_mean(compression):.1f}x", _num(_mean([run.get("state_items") for run in runs])),
            sum(1 for run in runs for t in run["turns"] if t.get("error"))]


def ladder_section(ladder: dict) -> list[str]:
    """The verification ladder: unsupported-claim rate and fact recall per mode."""

    rows = []

    for mode, s in ladder["summary"].items():
        mine = [r for r in ladder["rows"] if r.get("mode") == mode and "error" not in r]
        judged = [r for r in mine if r.get("unsupported") is not None]
        returned, unsupported = sum(r["returned"] for r in judged), sum(r["unsupported"] for r in judged)
        low, high = S.exact_interval(unsupported, returned) if returned else (None, None)
        recalls = [r["fact_recall"] for r in mine if r.get("fact_recall") is not None]
        rows.append([f"`{mode}`", "-" if not returned else f"{unsupported}/{returned} = {100 * unsupported / returned:.1f}% ({100 * low:.0f}-{100 * high:.0f})",
                     "-" if not recalls else f"{100 * _mean(recalls):.0f}% (n={len(recalls)})", _num(s["avg_memory_tokens"]), s["sentences_returned"], s["sentences_dropped"]])

    return ["## The verification ladder (C1)", "",
            f"For the memory probes of the `{ladder.get('variant', 'full')}` conversation ({ladder.get('turns', '?')} turns), each mode rebuilds the memory and a separate judge model "
            f"(`{ladder.get('judge_model', '?')}`; reconstruction model `{ladder.get('reconstruction_model', '?')}`) scores what would go into the prompt. "
            "*Unsupported* = returned sentences that neither the archive nor the anchored pages support; *fact recall* = reference facts (from the archive) the memory states.", "",
            _table(["Mode", "Unsupported sentences (exact 95% CI)", "Fact recall", "Memory tokens", "Sentences returned", "Sentences dropped"], rows), ""]


def audit_section(audit: dict) -> list[str]:
    """The verifier's own error rate, from the hand-labelled sheet."""

    def cell(r):
        return "-" if not r["n"] else f"{r['errors']}/{r['n']} = {100 * r['rate']:.1f}% (exact 95% interval {100 * r['interval'][0]:.0f}-{100 * r['interval'][1]:.0f}%)"

    return ["## How often the verifier is wrong (hand-labelled sample)", "",
            f"{audit['labelled']} sentences labelled by one person against their sources. **False accept**: a kept sentence the sources do not support (an unsupported claim reaches "
            "the prompt). **False reject**: a dropped sentence the sources do support (information is lost).", "",
            _table(["Error", "Rate"], [["false accept", cell(audit["false_accept"])], ["false reject", cell(audit["false_reject"])]]), ""]


def render(by_variant: dict, meta: dict, ladder: dict | None = None, audit: dict | None = None) -> str:

    names = list(by_variant)
    out = ["# SnapRAG-LongConv -- long-conversation results", "",
           f"Document: **{meta.get('document', 'cu31924019203078.pdf')}** | LLM: **{meta.get('model', '?')}** | {meta.get('turns', '?')} turns, seed {meta.get('seed', 0)}, "
           f"{meta.get('repeats', 1)} repeat(s) | {meta.get('date', '')} | commit `{meta.get('commit', '?')}`", "",
           "Probes are scored by fixed patterns, not by an LLM judge. Each cell is k/n with an exact (Clopper-Pearson) 95% interval; a probe whose source turn "
           "failed to run is excluded. *Memory probes* are recall, supersede and reference; *memory loss* is the accuracy on near probes minus that on very-far probes.", "",
           "## Probe accuracy", ""]

    header = ["Variant", "All probes", "Memory probes"] + list(KINDS) + [f"{b} (memory)" for b in BUCKETS] + ["Memory loss"]
    rows = []

    for name in names:
        probes = probes_of(by_variant[name])
        memory = [p for p in probes if p["kind"] in MEMORY_KINDS and not p.get("excluded")]

        def acc(bucket):
            part = [p for p in memory if p.get("bucket") == bucket]
            return sum(p["correct"] for p in part) / len(part) if part else None

        loss = None if acc("near") is None or acc("very_far") is None else acc("near") - acc("very_far")
        rows.append([f"`{name}`", _cell(probes), _cell(memory)] + [_cell([p for p in probes if p["kind"] == k]) for k in KINDS]
                    + [_cell([p for p in memory if p.get("bucket") == b]) for b in BUCKETS] + ["-" if loss is None else f"{100 * loss:+.0f} pp"])

    out += [_table(header, rows), ""]

    out += ["## Cost per turn (provider-reported)", "",
            "Tokens and calls cover the answer and the memory upkeep of the turn. Active seconds exclude the wait for the request-rate limit "
            "(rows run before this was measured show a dash). Compression is the source tokens of a snapshot over its own tokens.", "",
            _table(["Variant", "Prompt tokens", "Completion tokens", "LLM calls (answer + upkeep)", "Active seconds (mean / p95)", "Doc page hit", "Snapshots", "Compression", "State items", "Errors"],
                   [[f"`{n}`"] + cost_row(by_variant[n]) for n in names]), ""]

    for ref in REFERENCES:
        if ref not in by_variant:
            continue

        table = {n: paired(by_variant[n], by_variant[ref]) for n in names if n != ref}
        table = {n: d for n, d in table.items() if d}

        for n, p in S.holm({n: d["p"] for n, d in table.items()}).items():
            table[n]["p_adj"] = p

        if table:
            out += [f"## Each variant against `{ref}` (paired on the same probes)", "",
                    "Difference in accuracy (95% bootstrap interval); probes won, lost, tied; exact sign test p and Holm-adjusted p over the variants compared; "
                    "better or worse only when the adjusted p is below 0.05.", "",
                    _table(["Variant", "Difference", "Won / lost / tied", "p", "Holm p", "Verdict"],
                           [[f"`{n}`", f"{100 * d['mean']:+.1f} pp ({100 * d['ci'][0]:+.0f} to {100 * d['ci'][1]:+.0f})", f"{d['wins']} / {d['losses']} / {d['n'] - d['wins'] - d['losses']}",
                             f"{d['p']:.3f}", f"{d['p_adj']:.3f}", ("better" if d["mean"] > 0 else "worse") if d["p_adj"] < ALPHA else "no clear difference"]
                            for n, d in table.items()]), ""]

    out += ["## Why probes failed", ""]

    for name in names:
        reasons = {}
        for p in probes_of(by_variant[name]):
            if not p["correct"] and not p.get("excluded"):
                key = p["reason"].split(":")[0]
                reasons[key] = reasons.get(key, 0) + 1
        out.append(f"- `{name}`: " + (", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) or "none"))

    out.append("")

    if ladder:
        out += ladder_section(ladder)

    if audit:
        out += audit_section(audit)

    return "\n".join(out)


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--ladder")
    parser.add_argument("--audit", help="a labelled sheet from benchmarks.verifier_audit")
    parser.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results", "longconv_report.md"))
    parser.add_argument("--pdf")
    parser.add_argument("--latex")
    args = parser.parse_args()

    by_variant, meta = pool(args.results)
    ladder = json.load(open(args.ladder, encoding="utf-8")) if args.ladder and os.path.exists(args.ladder) else None
    audit = None

    if args.audit and os.path.exists(args.audit):
        from benchmarks.verifier_audit import score
        audit = score(json.load(open(args.audit, encoding="utf-8")))

    text = render(by_variant, meta, ladder, audit)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)

    print(args.out)

    if args.pdf:
        print(write_pdf(text, args.pdf))

    if args.latex:
        with open(args.latex, "w", encoding="utf-8") as f:
            f.write(to_latex(text))
        print(args.latex)


if __name__ == "__main__":
    main()
