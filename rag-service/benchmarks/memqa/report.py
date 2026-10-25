"""Markdown tables from the memqa result files: python -m benchmarks.memqa.report [files...] > benchmarks/results/memqa_report.md"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.docqa.stats import exact_interval, holm, sign_p  # noqa: E402

# Third-party list price for Gemini 3.5 Flash-Lite, US$ per million tokens (not checked against Google's page).
PRICE_IN, PRICE_OUT = 0.30, 2.50


def pct(k: int, n: int) -> str:

    if not n:
        return "-"

    low, high = exact_interval(k, n)

    return f"{k}/{n} = {k / n:.1%} ({low * 100:.0f}-{high * 100:.0f})"


def render(path: str) -> str:

    data = json.load(open(path, encoding="utf-8"))
    meta, runs = data["meta"], data["runs"]
    systems = list(dict.fromkeys(r["system"] for r in runs))
    rows = {s: [row for r in runs if r["system"] == s for row in r["rows"] if not row["error"]] for s in systems}
    errors = {s: sum(1 for r in runs if r["system"] == s for row in r["rows"] if row["error"]) for s in systems}
    out = [f"### {meta['dataset']}  ({meta['date']}, model {meta['model']}, commit {meta['commit']}, seed {meta['seed']})\n"]

    out += ["| System | Correct (95% exact interval) | Answerable: correct | F1 (answerable) | Abstention: declined when it should | Failed calls |", "|---|---|---|---|---|---|"]

    for s in systems:
        mine, answerable = rows[s], [x for x in rows[s] if not x["abstain"]]
        abstain = [x for x in rows[s] if x["abstain"]]
        f1s = [x["f1"] for x in answerable if x["f1"] is not None]
        out.append(f"| `{s}` | {pct(sum(x['correct'] for x in mine), len(mine))} | {pct(sum(x['correct'] for x in answerable), len(answerable))} | "
                   f"{(sum(f1s) / len(f1s)) if f1s else 0:.3f} | {pct(sum(x['correct'] for x in abstain), len(abstain))} | {errors[s]} |")

    types = sorted({x["type"] for s in systems for x in rows[s]})
    out += ["", "By question type (correct / asked):", "", "| Type | " + " | ".join(f"`{s}`" for s in systems) + " |", "|---|" + "---|" * len(systems)]

    for t in types:
        cells = []
        for s in systems:
            mine = [x for x in rows[s] if x["type"] == t]
            cells.append(f"{sum(x['correct'] for x in mine)}/{len(mine)}")
        out.append(f"| {t} | " + " | ".join(cells) + " |")

    out += ["", "Cost per question and cost of building the memory (provider-reported tokens; seconds exclude rate-limit waits):", "",
            "| System | Query tokens in/out | Query LLM calls | Active s/question | Build calls | Build tokens in/out | Snapshots | Est. US$ per 100 questions* |", "|---|---|---|---|---|---|---|---|"]

    for s in systems:
        mine, mine_runs = rows[s], [r for r in runs if r["system"] == s]
        n = max(1, len(mine))
        tin, tout = sum(x["tokens_in"] for x in mine) / n, sum(x["tokens_out"] for x in mine) / n
        b_in, b_out = sum(r["build_tokens_in"] for r in mine_runs), sum(r["build_tokens_out"] for r in mine_runs)
        queries = max(1, len(mine))
        dollars = 100 * ((tin * PRICE_IN + tout * PRICE_OUT) / 1e6 + (b_in * PRICE_IN + b_out * PRICE_OUT) / 1e6 / queries)
        out.append(f"| `{s}` | {tin:,.0f} / {tout:,.0f} | {sum(x['calls'] for x in mine) / n:.1f} | {sum(x['active_seconds'] for x in mine) / n:.1f} | "
                   f"{sum(r['build_calls'] for r in mine_runs)} | {b_in:,} / {b_out:,} | {sum(r['snapshots'] for r in mine_runs)} | {dollars:.2f} |")

    out.append("\n*list price assumed at $0.30 / $2.50 per million input / output tokens, build cost spread over the questions asked; not checked against Google's page.")

    if "no_memory" in systems and len(systems) > 1:
        key = lambda x: (x["id"], x["question"])  # noqa: E731
        base = {key(x): x["correct"] for x in rows["no_memory"]}
        tests = {}
        for s in systems:
            if s == "no_memory":
                continue
            shared = [(x["correct"], base[key(x)]) for x in rows[s] if key(x) in base]
            wins, losses = sum(a and not b for a, b in shared), sum(b and not a for a, b in shared)
            tests[s] = (wins, losses, sign_p(wins, losses))
        adjusted = holm({s: t[2] for s, t in tests.items()})
        out += ["", "Paired against `no_memory` (exact sign test, Holm-adjusted over these tests):", "", "| System | Wins | Losses | p | Holm p |", "|---|---|---|---|---|"]
        out += [f"| `{s}` | {w} | {l} | {p:.3f} | {adjusted[s]:.3f} |" for s, (w, l, p) in tests.items()]

    return "\n".join(out)


def main() -> None:

    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "benchmarks", "results", "memqa_*.json")))
    print("## Conversational-memory QA (LoCoMo, LongMemEval)\n")
    print("Sampled questions, lexical scoring (token F1 >= 0.5 or the gold answer contained); a small sample, so read the intervals, not the point estimates.\n")

    for path in paths:
        if "smoke" not in path:
            print(render(path) + "\n")


if __name__ == "__main__":
    main()
