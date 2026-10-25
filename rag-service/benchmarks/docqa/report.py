"""Turns docqa results files into a Markdown report (and a PDF and LaTeX tables).

    python -m benchmarks.docqa.report benchmarks/results/docqa_<time>.json [--pdf out.pdf] [--latex out.tex]

Proportions come with exact (Clopper-Pearson) 95% intervals, scores with bootstrap
intervals. Architectures are compared with SnapRAG on the same questions (paired):
an exact sign test for right/wrong, a Wilcoxon signed-rank test for scores, and
Holm-Bonferroni adjustment across the whole family of comparisons.
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from benchmarks.docqa import stats as S
from benchmarks.docqa.export import to_latex, write_pdf

REFERENCE = "dense_k5"          # the naive RAG the "vs naive" column is measured against
HEADLINE = "snaprag"            # the system compared with every other one in the paired table
TYPES = ("detail", "numeric", "aggregation", "multi_hop", "inference")
PUBLISHED = ("hyde", "crag", "self_rag", "ircot", "adaptive_rag", "abrag")
# Diagnostics are not competitors: they feed the planner ablation and the calibration.
DIAGNOSTIC = ("snaprag_fixed_k_noplan", "snaprag_retrieval", "snaprag_retrieval_planner", "snaprag_calib")
ALPHA = 0.05


# ---------- small helpers ----------

def _mean(values):

    values = [v for v in values if v is not None]

    return sum(values) / len(values) if values else None


def _pct(value, digits=1):

    return "-" if value is None else f"{100 * value:.{digits}f}%"


def _num(value, fmt="{:.1f}"):

    return "-" if value is None else fmt.format(value)


def _percentile(values, q):

    values = sorted(values)

    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def _pp(value):

    return "-" if value is None else f"{100 * value:+.1f} pp"


def _cell(values, bold=False) -> str:
    """A proportion as 'k/n = p% (exact 95% interval)', or a mean score with a bootstrap interval."""

    values = [float(v) for v in values if v is not None]

    if not values:
        return "-"

    if S.is_binary(values):
        k, n = int(sum(values)), len(values)
        low, high = S.exact_interval(k, n)
        text = f"{k}/{n} = {_pct(k / n)} ({100 * low:.0f}-{100 * high:.0f})"
    else:
        low, high = S.bootstrap_ci(values)
        text = f"{_pct(sum(values) / len(values))} ({100 * low:.0f}-{100 * high:.0f})"

    return f"**{text}**" if bold else text


def _count(values) -> str:
    """'k/n (p%)' for right/wrong outcomes."""

    return "-" if not values else f"{int(sum(values))}/{len(values)} ({_pct(sum(values) / len(values), 0)})"


def _table(header, rows):

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]

    return "\n".join(lines)


def _best(values: dict):
    """Names holding the highest value (ties included); none when all are equal."""

    present = {k: v for k, v in values.items() if v is not None}

    if len(set(present.values())) < 2:
        return set()

    return {k for k, v in present.items() if v == max(present.values())}


def _by_arch(rows):

    grouped = {}

    for row in rows:
        grouped.setdefault(row["arch"], []).append(row)

    return grouped


def _answered(rows):
    return [r for r in rows if "error" not in r and "correct" in r]


# ---------- paired comparison ----------

def _delta(a: dict, b: dict, key: str):
    """Paired difference a - b over the questions both have: a dict with the mean, its
    bootstrap interval, wins / losses / ties and the raw p-value; None without shared questions."""

    common = [i for i in a if i in b and a[i].get(key) is not None and b[i].get(key) is not None]

    if not common:
        return None

    diffs = [float(a[i][key]) - float(b[i][key]) for i in common]

    return {"mean": sum(diffs) / len(diffs), "ci": S.bootstrap_ci(diffs), "wins": sum(d > 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs), "n": len(diffs), "p": S.paired_p(diffs)}


def _pairs(headline, names, grouped):
    """{arch: {metric: delta}} of the headline against every other architecture, with
    Holm-adjusted p-values added per metric across the whole family."""

    def rows_of(name):
        return {r["id"]: r for r in grouped[name] if "error" not in r}

    head = rows_of(headline)
    table = {}

    for n in names:
        if n != headline:
            other = rows_of(n)
            table[n] = {metric: _delta(head, other, metric) for metric in ("correct", "score", "context_recall")}

    for metric in ("correct", "score", "context_recall"):
        adjusted = S.holm({n: d[metric]["p"] for n, d in table.items() if d[metric]})
        for n, p in adjusted.items():
            table[n][metric]["p_adj"] = p

    return table


def _verdict(d) -> str:

    if d is None:
        return "not enough shared questions"

    if d["p_adj"] < ALPHA and d["ci"][0] * d["ci"][1] > 0:
        return "better" if d["mean"] > 0 else "worse"

    return "no clear difference"


def _show(d) -> str:

    if d is None:
        return "-"

    low, high = d["ci"]

    return (f"{_pp(d['mean'])} ({100 * low:+.0f} to {100 * high:+.0f}); won {d['wins']}, lost {d['losses']}, tied {d['n'] - d['wins'] - d['losses']}; "
            f"p={d['p']:.3f}, Holm {d['p_adj']:.3f}; **{_verdict(d)}**")


# ---------- the report ----------

def _summary(names, stats, grouped, pairs) -> list[str]:
    """The result in a few lines, from the numbers below."""

    if HEADLINE not in grouped:
        return []

    ranked = sorted((n for n in names if stats[n]["all"]), key=lambda n: -_mean(stats[n]["all"]))
    published = [n for n in PUBLISHED if n in stats and stats[n]["all"]]
    out = ["## Summary", "",
           "Accuracy ranking: " + "; ".join(f"`{n}` {_pct(_mean(stats[n]['all']), 0)}" for n in ranked) + ".", ""]

    def line(prefix, name):
        d = pairs.get(name, {}).get("correct")
        return f"- {prefix}: {_pct(_mean(stats[HEADLINE]['all']))} against {_pct(_mean(stats[name]['all']))}, {_show(d)}."

    if published:
        top = max(published, key=lambda n: _mean(stats[n]["all"]))
        out.append(line(f"SnapRAG against the best published method re-implemented here (`{top}`)", top))

    for name, what in (("snaprag_fixed_k", "SnapRAG without the extra AB-RAG batches (`snaprag_fixed_k`), i.e. the effect of the adaptive batches"),
                       ("abrag", "AB-RAG with one fixed first batch and budget for every question (`abrag`), i.e. the effect of the intent-aware batch"),
                       ("hybrid_rerank_k5", "a standard hybrid + cross-encoder pipeline (`hybrid_rerank_k5`)")):
        if name in stats and stats[name]["all"] and name in pairs:
            out.append(line(f"Against {what}", name))

    if stats[HEADLINE]["calls"]:
        out.append(f"- Cost: `{HEADLINE}` makes {_num(_mean(stats[HEADLINE]['calls']))} LLM calls per question; "
                   + ", ".join(f"`{n}` {_num(_mean(stats[n]['calls']))}" for n in published if stats[n]["calls"]) + ".")

    return out + [""]


def render(result: dict, dataset: dict | None = None) -> str:

    meta = result["meta"]
    ids = meta.get("ids")
    grouped = _by_arch([r for r in result["rows"] if ids is None or r["id"] in ids])
    everything = [n for n in meta.get("archs", grouped) if n in grouped]
    names = [n for n in everything if n not in DIAGNOSTIC]
    diagnostics = [n for n in everything if n in DIAGNOSTIC]
    descriptions, sources = meta.get("archs", {}), meta.get("sources", {})
    retrieval_only = meta.get("retrieval_only", False)
    questions = {q["id"]: q for q in (dataset or {}).get("questions", [])}
    used = [q for q in questions.values() if ids is None or q["id"] in ids]
    total = len(questions) or meta["questions"]

    out = [f"# Document-QA architecture comparison -- {'retrieval only (no LLM answers)' if retrieval_only else 'answers'}", "",
           f"Document: **{meta['document']}** | LLM: **{'none' if retrieval_only else meta['model']}** | {meta['questions']} questions | "
           f"run {meta['started']}" + (f" to {meta['finished']}" if meta.get("finished") else "")
           + (f" | commit `{meta['commit']}`" if meta.get("commit") else "")
           + (f" | seed {meta['seed']}, temperature {meta['temperature']}" if "seed" in meta else ""), ""]

    # ---- setup
    out += ["## Architectures", "", _table(["Name", "What it is", "Source"], [[f"`{n}`", descriptions.get(n, ""), sources.get(n, "")] for n in names]), ""]

    if diagnostics:
        out += ["Diagnostics (not competitors; used in the ablation and calibration sections):", "",
                _table(["Name", "What it is"], [[f"`{n}`", descriptions.get(n, "")] for n in diagnostics]), ""]

    counts = {}
    for row in grouped[everything[0]]:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    unans = sum(1 for r in grouped[everything[0]] if not r["answerable"])
    shared = ("index, embedder and cross-encoder; no LLM answers are generated, and SnapRAG's planner and HyDE run their offline fallbacks"
              if retrieval_only else "index, embedder, cross-encoder, answer prompt and LLM")
    out += [f"All architectures use the same {shared}. Questions: "
            + ", ".join(f"{n} {t}" for t, n in sorted(counts.items())) + f" ({unans} unanswerable or false-premise). "
            "Scoring is by fixed patterns, not by an LLM judge.", ""]

    # ---- dataset
    if used:
        pages = [p for q in used for p in q["pages"]]
        kinds_all = sorted({q["type"] for q in used})
        out += ["## Dataset", "",
                f"{len(used)} questions written by reading the book; every answerable question lists the facts an answer needs and the "
                f"gold pages that hold them ({len(set(pages))} distinct pages, from page {min(pages)} to {max(pages)}). "
                "Multi-hop questions need facts from two or more pages that are not neighbours."]

        if len(used) < total:
            step = total / len(used)
            out += [f"This is an evenly spaced sample of {len(used)} of the dataset's {total} questions (question i of the sample is question "
                    f"floor(i x {step:.2f}) of the dataset, `--sample {len(used)}`). It was chosen by position, not by any result, only to fit "
                    "the free-tier request limit, and is the same for every architecture."]

        out += ["", _table(["Question type", "Easy", "Medium", "Hard", "Total"],
                           [[t] + [sum(1 for q in used if q["type"] == t and q["difficulty"] == d) for d in ("easy", "medium", "hard")]
                            + [sum(1 for q in used if q["type"] == t)] for t in kinds_all]
                           + [["**all**"] + [sum(1 for q in used if q["difficulty"] == d) for d in ("easy", "medium", "hard")] + [len(used)]]), ""]

    out += ["## Metrics", "",
            "- **Accuracy**: share of questions answered correctly, shown as k/n with an exact (Clopper-Pearson) 95% interval. An answerable question is correct "
            "when the answer contains every required fact (fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer "
            "declines or states the correction.",
            "- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions; bootstrap interval).",
            "- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).",
            "- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).",
            "- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.",
            "- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.",
            "- **LLM calls, tokens, pages read, seconds**: cost of one question. Tokens are the counts the provider reported (prompt and completion, all calls of the "
            "question); seconds are shown without the time spent waiting for the request-rate limit, and with it (wall).",
            "- **Confidence check**: the AB-RAG checker's verdict on an answer (not the reconstruction check of the memory method); AUROC is the chance that a right "
            "answer has a higher confidence than a wrong one (0.5 = the confidence says nothing).",
            "- **Comparisons** with SnapRAG are paired on the same questions: an exact two-sided sign test for right/wrong, a Wilcoxon signed-rank test for scores, "
            f"Holm-Bonferroni adjustment across all {max(len(names) - 1, 0)} comparisons of each metric, and a percentile bootstrap interval (2000 resamples, seed 0) of the "
            "difference. A difference is called better or worse only when the adjusted p-value is below 0.05 and the interval excludes zero.", ""]

    if meta.get("settings"):
        out += ["## Run settings", "", _table(["Setting", "Value"], [[f"`{k}`", v] for k, v in meta["settings"].items()]), ""]

    # ---- per-architecture statistics
    stats = {}
    for name in everything:
        rows = grouped[name]
        done = _answered(rows)
        ans = [r for r in rows if r["answerable"] and "error" not in r]
        stats[name] = {
            "errors": sum("error" in r for r in rows),
            "all": [float(r["correct"]) for r in done],
            "answerable": [float(r["correct"]) for r in done if r["answerable"]],
            "unanswerable": [float(r["correct"]) for r in done if not r["answerable"]],
            "fact": [r["score"] for r in done if r["answerable"]],
            "refusal": [float(r["declined"]) for r in done if r["answerable"]],
            "ctx": [r["context_recall"] for r in ans if "context_recall" in r],
            "page_hit": [float(r["page_hit"]) for r in ans if "page_hit" in r],
            "page_recall": [r["page_recall"] for r in ans if "page_recall" in r],
            "passages": [r["passages"] for r in rows if "passages" in r],
            "chars": [r["context_chars"] for r in rows if "context_chars" in r],
            "calls": [r["calls"] for r in rows if "calls" in r],
            "tokens_in": [r["tokens_in"] for r in rows if r.get("tokens_in")],
            "tokens_out": [r["tokens_out"] for r in rows if r.get("tokens_out")],
            "seconds": [r["seconds"] for r in rows if "seconds" in r],
            "active": [r["active_seconds"] for r in rows if "active_seconds" in r],
        }

    answering = [n for n in names if stats[n]["all"]]
    pairs = _pairs(HEADLINE if HEADLINE in answering else "snaprag_fixed_k", answering, grouped) if len(answering) > 1 and (HEADLINE in answering or "snaprag_fixed_k" in answering) else {}
    headline = HEADLINE if HEADLINE in answering else "snaprag_fixed_k"

    # ---- answer quality
    if answering:
        short = [n for n in answering if len(stats[n]["all"]) < meta["questions"]]

        if short:
            out += ["> **Partial results:** " + ", ".join(f"`{n}`" for n in short) + " have not answered every question (the run is unfinished or hit a rate limit), "
                    "so their numbers rest on fewer questions.", ""]

        out += _summary(answering, stats, grouped, pairs) if HEADLINE in answering else []

        best = _best({n: _mean(stats[n]["all"]) for n in answering})
        ref = _mean(stats[REFERENCE]["all"]) if REFERENCE in stats else None
        out += ["## Answer quality", "",
                _table(["Architecture", "Accuracy, all questions (exact 95% CI)", f"vs `{REFERENCE}`", "Answerable only", "Mean fact score",
                        "Unanswerable handled", "Refused an answerable question"],
                       [[f"`{n}`", _cell(stats[n]["all"], n in best),
                         "-" if n == REFERENCE or ref is None else _pp(_mean(stats[n]["all"]) - ref),
                         _cell(stats[n]["answerable"]), _cell(stats[n]["fact"]), _count(stats[n]["unanswerable"]),
                         _count(stats[n]["refusal"])] for n in answering]), ""]

    # ---- retrieval quality
    retrievers = [n for n in everything if any(p > 0 for p in stats[n]["passages"])]     # not the no-retrieval baseline
    main_retrievers = [n for n in retrievers if n not in DIAGNOSTIC]
    out += ["## Retrieval quality", "",
            "Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. "
            "These are measured on answerable questions only.", "",
            _table(["Architecture", "Context recall (95% CI)", "Page hit (exact 95% CI)", "Gold pages read", "Pages read (mean)", "Context (characters)"],
                   [[f"`{n}`", _cell(stats[n]["ctx"], n in _best({m: _mean(stats[m]["ctx"]) for m in main_retrievers})), _cell(stats[n]["page_hit"]),
                     _pct(_mean(stats[n]["page_recall"])), _num(_mean(stats[n]["passages"])), _num(_mean(stats[n]["chars"]), "{:,.0f}")]
                    for n in main_retrievers]), ""]

    kinds = [t for t in TYPES if any(r["type"] == t and "context_recall" in r for r in grouped[everything[0]])]
    sizes = [sum(1 for r in grouped[everything[0]] if r["type"] == t and "context_recall" in r) for t in kinds]
    out += ["### Context recall by question type", "",
            _table(["Architecture"] + [f"{t} (n={s})" for t, s in zip(kinds, sizes)],
                   [[f"`{n}`"] + [_pct(_mean([r["context_recall"] for r in grouped[n] if r["type"] == t and "context_recall" in r])) for t in kinds]
                    for n in main_retrievers]), ""]

    # ---- the planner ablation
    ablation = [n for n in ("snaprag_fixed_k_noplan", "snaprag_retrieval_planner", "snaprag_retrieval") if n in grouped and stats[n]["ctx"]]
    if len(ablation) >= 2:
        label = {"snaprag_fixed_k_noplan": "heuristic plan, no HyDE (no LLM before retrieval)", "snaprag_retrieval_planner": "LLM planner, no HyDE",
                 "snaprag_retrieval": "LLM planner + HyDE (SnapRAG's default)"}
        base = {r["id"]: r for r in grouped[ablation[0]] if "error" not in r}
        diffs = {n: _delta({r["id"]: r for r in grouped[n] if "error" not in r}, base, "context_recall") for n in ablation[1:]}
        adjusted = S.holm({n: d["p"] for n, d in diffs.items() if d})
        out += ["## Ablation: what the planner and HyDE do to retrieval", "",
                "The same first batch of pages SnapRAG reads, retrieved three ways. Differences are paired against the first row (Wilcoxon signed-rank on context recall, Holm-adjusted).", "",
                _table(["Retrieval", "Context recall (95% CI)", "Page hit", "Gold pages read", "Pages read (mean)", "Difference in context recall vs first row"],
                       [[label[n], _cell(stats[n]["ctx"]), _cell(stats[n]["page_hit"]), _pct(_mean(stats[n]["page_recall"])), _num(_mean(stats[n]["passages"])),
                         "-" if n == ablation[0] or not diffs.get(n) else f"{_pp(diffs[n]['mean'])} ({100 * diffs[n]['ci'][0]:+.0f} to {100 * diffs[n]['ci'][1]:+.0f}); "
                         f"p={diffs[n]['p']:.3f}, Holm {adjusted[n]:.3f}"] for n in ablation]), ""]

    # ---- cost
    if answering:
        def per_call(n):
            calls = _mean(stats[n]["calls"])
            return "-" if not calls or not stats[n]["all"] else f"{100 * _mean(stats[n]['all']) / calls:.1f} pp"

        out += ["## Cost and speed", "",
                "Tokens are the provider's own counts, summed over all calls of a question (rows run before token counting was added show a dash). "
                "Active seconds exclude the time the client waited for the request-rate limit; wall seconds include it, so they depend on the rate limit, not the method.", "",
                _table(["Architecture", "LLM calls", "Prompt tokens", "Completion tokens", "Pages read", "Accuracy per call",
                        "Active seconds (median / p95)", "Wall seconds (median)", "Errors"],
                       [[f"`{n}`", _num(_mean(stats[n]["calls"])), _num(_mean(stats[n]["tokens_in"]), "{:,.0f}"), _num(_mean(stats[n]["tokens_out"]), "{:,.0f}"),
                         _num(_mean(stats[n]["passages"])), per_call(n),
                         "-" if not stats[n]["active"] else f"{_num(_percentile(stats[n]['active'], 0.5))} / {_num(_percentile(stats[n]['active'], 0.95))}",
                         _num(_percentile(stats[n]["seconds"], 0.5)), stats[n]["errors"]] for n in answering]), ""]

        # ---- the confidence check
        checked = [n for n in answering if any("supported" in r for r in grouped[n])]
        if checked:
            def split(n, flag):
                return [float(r["correct"]) for r in _answered(grouped[n]) if r.get("supported") is flag]

            def confidence_auroc(n):
                rows = [r for r in _answered(grouped[n]) if r.get("confidence") is not None]
                value = S.auroc([r["confidence"] for r in rows], [int(r["correct"]) for r in rows])
                return "-" if value is None else f"{value:.2f}"

            out += ["## Confidence check", "",
                    "The AB-RAG checker judges whether the retrieved text supports the answer and gives a confidence; an answer below the stopping threshold "
                    "triggers another batch of pages. A useful check separates right from wrong answers: accuracy should be higher among the answers that pass, and AUROC "
                    "should be well above 0.5.", "",
                    _table(["Architecture", "Mean confidence", "Passed the check", "Accuracy when passed", "Accuracy when not passed", "AUROC of confidence"],
                           [[f"`{n}`", _num(_mean([r.get("confidence") for r in grouped[n]]), "{:.2f}"),
                             _count([float(r["supported"]) for r in grouped[n] if "supported" in r]), _cell(split(n, True)), _cell(split(n, False)),
                             confidence_auroc(n)] for n in checked]), ""]

        for n in [n for n in answering if any("route" in r for r in grouped[n])]:
            share = {k: sum(1 for r in grouped[n] if r.get("route") == k) for k in "ABC"}
            out += [f"`{n}` routed {share['A']} questions to no retrieval, {share['B']} to single-step retrieval and {share['C']} to multi-step retrieval.", ""]

        # ---- by type and difficulty
        for title, key, groups in (("Accuracy by question type", "type", TYPES + ("false_premise", "unanswerable")),
                                   ("Accuracy by difficulty", "difficulty", ("easy", "medium", "hard"))):
            present = [g for g in groups if any(r[key] == g for r in grouped[answering[0]])]
            table = [[f"`{n}`"] + [_count([float(r["correct"]) for r in _answered(grouped[n]) if r[key] == g]) for g in present] for n in answering]
            sizes = [sum(1 for r in grouped[answering[0]] if r[key] == g) for g in present]
            out += [f"## {title}", "", _table(["Architecture"] + [f"{g} (n={s})" for g, s in zip(present, sizes)], table), ""]

    # ---- paired comparison
    if pairs:
        out += [f"## `{headline}` against every other architecture (paired, same questions)", "",
                "Each cell: mean difference in percentage points (95% bootstrap interval); questions won, lost and tied; raw p-value and Holm-adjusted p-value "
                f"over the {len(pairs)} comparisons; the verdict.", "",
                _table(["Compared with", "Accuracy difference", "Fact-score difference", "Context-recall difference"],
                       [[f"`{n}`", _show(d["correct"]), _show(d["score"]), _show(d["context_recall"])] for n, d in pairs.items()]), ""]
    elif retrieval_only and "snaprag_fixed_k" in grouped:
        pairs = _pairs("snaprag_fixed_k", main_retrievers, grouped)
        out += ["## `snaprag_fixed_k` against every other architecture (paired, same questions; retrieval only)", "",
                _table(["Compared with", "Context-recall difference"], [[f"`{n}`", _show(d["context_recall"])] for n, d in pairs.items()]), ""]

    # ---- failures
    if HEADLINE in answering:
        wrong = [r for r in _answered(grouped[HEADLINE]) if not r["correct"]]
        if wrong:
            out += [f"## Questions `{HEADLINE}` got wrong ({len(wrong)} of {len(stats[HEADLINE]['all'])})", ""]
            for r in wrong:
                item = questions.get(r["id"], {})
                out.append(f"- **{r['id']}** ({r['type']}, {r['difficulty']}): {item.get('question', '')}  \n"
                           f"  expected: {item.get('answer', '')}  \n  answered: {' '.join(str(r.get('answer', '')).split())[:260]}")
            out.append("")

    out += ["## Notes", "",
            f"- One document, one LLM and {meta['questions']} questions written by reading the book: the intervals show how little this many questions can "
            "separate close systems. Treat differences that are not marked better or worse as noise.",
            "- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; "
            "they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.",
            "- The book is a scanned novel; some pages have broken text, which lowers every architecture equally. It is under copyright, so neither the PDF nor "
            "its page text is released; the questions, answers and gold page numbers are.",
            "- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.",
            "- Scoring requires every fact in the answer, so an answer that is right but leaves one fact out counts as wrong; the fact score shows the partial credit.", ""]

    return "\n".join(out)


def write_report(result: dict, path: str) -> str:

    with open(path, "w", encoding="utf-8") as f:
        f.write(render(result, _dataset()))

    return path


def _dataset():

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")

    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def run_settings() -> dict:
    """The settings that shape the results, from the current configuration (for the reproducibility table)."""

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, "..", ".env"))
        from app.core.config import settings
    except Exception:
        return {}

    keys = ("LLM_MODEL", "LLM_REASONING_EFFORT", "EMBEDDING_MODEL", "CROSS_ENCODER_MODEL", "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE",
            "STOP_CONFIDENCE", "MAX_EXTRA_BATCHES", "CONF_W_MODEL", "CONF_W_SIMILARITY", "CONF_W_SPREAD", "LLM_REQUESTS_PER_MINUTE")

    return {key: getattr(settings, key) for key in keys} | {"temperature": 0}


def prepare(results: list[dict]) -> None:
    """Fills in what older result files lack: the run settings and each architecture's description and source."""

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, "..", ".env"))
        from benchmarks.docqa.architectures import ARCHITECTURES
    except Exception:
        ARCHITECTURES = {}

    for result in results:
        meta = result["meta"]
        meta.setdefault("settings", run_settings())
        meta["archs"] = {**{n: v[0] for n, v in ARCHITECTURES.items() if any(r["arch"] == n for r in result["rows"])}, **meta.get("archs", {})}
        meta["sources"] = {**{n: v[3] for n, v in ARCHITECTURES.items()}, **meta.get("sources", {})}


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results files; several are put one after the other in one report")
    parser.add_argument("--out", help="Markdown file (default: next to the first results file)")
    parser.add_argument("--extra", action="append", default=[], help="Markdown file appended as it is (for example the calibration section); repeatable")
    parser.add_argument("--pdf", help="also write the report as a PDF")
    parser.add_argument("--latex", help="also write every table as LaTeX (booktabs) for a paper")
    args = parser.parse_args()

    dataset = _dataset()
    results = [json.load(open(path, encoding="utf-8")) for path in args.results]
    prepare(results)

    text ="\n\n---\n\n".join([render(result, dataset) for result in results] + [open(path, encoding="utf-8").read() for path in args.extra])
    out = args.out or os.path.splitext(args.results[0])[0] + ".md"

    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    print(out)

    if args.pdf:
        print(write_pdf(text, args.pdf))

    if args.latex:
        with open(args.latex, "w", encoding="utf-8") as f:
            f.write(to_latex(text))
        print(args.latex)


if __name__ == "__main__":
    main()
