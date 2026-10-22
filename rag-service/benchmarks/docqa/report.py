"""Turns a docqa results file into a Markdown report.

    python -m benchmarks.docqa.report benchmarks/results/docqa_<time>.json [--out report.md]

Every accuracy comes with a bootstrap 95% confidence interval over the questions,
and architectures are compared on the same questions (paired), so a difference
can be judged against the noise.
"""

import argparse
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from benchmarks.longconv.report import bootstrap_ci

REFERENCE = "dense_k5"          # the naive RAG the "vs naive" columns are measured against
HEADLINE = "snaprag"            # the system compared with every other one in the paired table
TYPES = ("detail", "numeric", "aggregation", "multi_hop", "inference")


def _mean(values):

    values = [v for v in values if v is not None]

    return sum(values) / len(values) if values else None


def _pct(value, digits=1):

    return "-" if value is None else f"{100 * value:.{digits}f}%"


def _cell(values, bold=False) -> str:
    """Mean of 0/1 or 0..1 values with its 95% interval, as a percentage."""

    values = [float(v) for v in values if v is not None]

    if not values:
        return "-"

    low, high = bootstrap_ci(values)
    text = f"{_pct(sum(values) / len(values))} ({100 * low:.0f}-{100 * high:.0f})"

    return f"**{text}**" if bold else text


def _num(value, fmt="{:.1f}"):

    return "-" if value is None else fmt.format(value)


def _percentile(values, q):

    values = sorted(values)

    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def _by_arch(rows):

    grouped = {}

    for row in rows:
        grouped.setdefault(row["arch"], []).append(row)

    return grouped


def _answered(rows):
    return [r for r in rows if "error" not in r and "correct" in r]


def _delta(a: dict, b: dict, key: str):
    """Paired difference a - b over the questions both answered:
    (mean, (low, high), wins, losses, n) or None."""

    common = [i for i in a if i in b and a[i].get(key) is not None and b[i].get(key) is not None]

    if not common:
        return None

    diffs = [float(a[i][key]) - float(b[i][key]) for i in common]

    return sum(diffs) / len(diffs), bootstrap_ci(diffs), sum(d > 0 for d in diffs), sum(d < 0 for d in diffs), len(diffs)


def _sign_p(wins: int, losses: int) -> float:
    """Exact two-sided sign-test p-value for wins vs losses (ties left out)."""

    n = wins + losses

    if n == 0:
        return 1.0

    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n

    return min(1.0, 2 * tail)


def _pp(value):
    return "-" if value is None else f"{100 * value:+.1f} pp"


def _table(header, rows):

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]

    return "\n".join(lines)


def _best(values: dict):
    """Names holding the highest value (ties included)."""

    present = {k: v for k, v in values.items() if v is not None}

    if len(set(present.values())) < 2:
        return set()

    return {k for k, v in present.items() if v == max(present.values())}


PUBLISHED = ("hyde", "crag", "self_rag", "ircot", "adaptive_rag", "abrag")


def _verdict(delta) -> str:

    if delta is None:
        return "not enough shared questions"

    mean, (low, high), wins, losses, size = delta
    word = "better" if low > 0 else "worse" if high < 0 else "no clear difference"

    return f"{_pp(mean)} (95% interval {100 * low:+.0f} to {100 * high:+.0f} pp; won {wins}, lost {losses}, tied {size - wins - losses} of {size} questions): {word}"


def _summary(names, stats, grouped) -> list[str]:
    """The result in a few lines, from the numbers below."""

    if HEADLINE not in grouped:
        return []

    def rows_of(name):
        return {r["id"]: r for r in grouped[name] if "error" not in r and "correct" in r}

    head = rows_of(HEADLINE)
    ranked = sorted((n for n in names if stats[n]["all"]), key=lambda n: -_mean(stats[n]["all"]))
    published = [n for n in PUBLISHED if n in stats and stats[n]["all"]]
    out = ["## Summary", "",
           "Accuracy ranking: " + "; ".join(f"`{n}` {_pct(_mean(stats[n]['all']), 0)}" for n in ranked) + ".", ""]

    if published:
        top = max(published, key=lambda n: _mean(stats[n]["all"]))
        out += [f"- SnapRAG against the best published method re-implemented here (`{top}`): "
                f"{_pct(_mean(stats[HEADLINE]['all']))} against {_pct(_mean(stats[top]['all']))}, {_verdict(_delta(head, rows_of(top), 'correct'))}."]

    for name, what in (("snaprag_fixed", "the extra AB-RAG batches (SnapRAG without them)"),
                       ("abrag", "the intent-aware first batch and budget (AB-RAG with one fixed batch for every question)"),
                       ("hybrid_rerank_k5", "a standard hybrid + cross-encoder pipeline")):
        if name in stats and stats[name]["all"]:
            out.append(f"- Against {what}: {_verdict(_delta(head, rows_of(name), 'correct'))}.")

    if "calls" in stats[HEADLINE] and stats[HEADLINE]["calls"]:
        out.append(f"- Cost: `{HEADLINE}` makes {_num(_mean(stats[HEADLINE]['calls']))} LLM calls per question; "
                   + ", ".join(f"`{n}` {_num(_mean(stats[n]['calls']))}" for n in published if stats[n]["calls"]) + ".")

    return out + [""]


def render(result: dict, dataset: dict | None = None) -> str:

    meta = result["meta"]
    ids = meta.get("ids")
    grouped = _by_arch([r for r in result["rows"] if ids is None or r["id"] in ids])
    names = [n for n in meta.get("archs", grouped) if n in grouped]
    descriptions = meta.get("archs", {})
    retrieval_only = meta.get("retrieval_only", False)
    questions = {q["id"]: q for q in (dataset or {}).get("questions", [])}

    out = ["# Document-QA architecture comparison", "",
           f"Document: **{meta['document']}** | LLM: **{'none (retrieval only)' if retrieval_only else meta['model']}** | "
           f"run {meta['started']} | {meta['questions']} questions", ""]

    # ---- setup
    out += ["## Architectures", "", _table(["Name", "What it is", "Source"], [[f"`{n}`", descriptions.get(n, ""), meta.get("sources", {}).get(n, "")] for n in names]), ""]

    counts = {}
    for row in grouped[names[0]]:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    unans = sum(1 for r in grouped[names[0]] if not r["answerable"])
    shared = ("index, embedder and cross-encoder; no LLM answers are generated, and SnapRAG's planner and HyDE run their offline fallbacks"
              if retrieval_only else "index, embedder, cross-encoder, answer prompt and LLM")
    out += [f"All architectures use the same {shared}. Questions: "
            + ", ".join(f"{n} {t}" for t, n in sorted(counts.items())) + f" ({unans} unanswerable or false-premise). "
            "Scoring is by fixed patterns, not by an LLM judge. Intervals are bootstrap 95% over questions.", ""]

    used = [q for q in questions.values() if ids is None or q["id"] in ids]

    if used:
        kinds_all = sorted({q["type"] for q in used})
        out += ["## Dataset", "",
                f"{len(used)} questions written by reading the book; every answerable question lists the facts an answer needs and the "
                f"gold pages that hold them ({len({p for q in used for p in q['pages']})} distinct pages, from page {min(p for q in used for p in q['pages'])} "
                f"to {max(p for q in used for p in q['pages'])}). Multi-hop questions need facts from two or more pages that are not neighbours.", "",
                _table(["Question type", "Easy", "Medium", "Hard", "Total"],
                       [[t] + [sum(1 for q in used if q["type"] == t and q["difficulty"] == d) for d in ("easy", "medium", "hard")]
                        + [sum(1 for q in used if q["type"] == t)] for t in kinds_all]
                       + [["**all**"] + [sum(1 for q in used if q["difficulty"] == d) for d in ("easy", "medium", "hard")] + [len(used)]]), ""]

    out += ["## Metrics", "",
            "- **Accuracy**: share of questions answered correctly. An answerable question is correct when the answer contains every required fact "
            "(fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer declines or states the correction.",
            "- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions).",
            "- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).",
            "- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).",
            "- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.",
            "- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.",
            "- **LLM calls, pages read, context tokens, seconds**: cost of one question.",
            "- **Intervals** are percentile bootstrap 95% intervals over questions (2000 resamples, seed 0); comparisons between architectures are "
            "paired on the same questions, with an exact two-sided sign test on the questions where they differ.", ""]

    if meta.get("settings"):
        out += ["## Run settings", "", _table(["Setting", "Value"], [[f"`{k}`", v] for k, v in meta["settings"].items()]), ""]

    stats = {}
    for name in names:
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
            "seconds": [r["seconds"] for r in rows if "seconds" in r],
        }

    # ---- answer quality
    if not retrieval_only:
        short = [n for n in names if len(stats[n]["all"]) < meta["questions"]]

        if short:
            out += ["> **Partial results:** " + ", ".join(f"`{n}`" for n in short) + " have not answered every question yet "
                    "(the run is unfinished or hit a rate limit), so their numbers rest on fewer questions.", ""]

        out += _summary(names, stats, grouped)

        best = _best({n: _mean(stats[n]["all"]) for n in names})
        ref = _mean(stats[REFERENCE]["all"]) if REFERENCE in stats else None
        out += ["## Answer quality", "",
                "Accuracy = every required fact is in the answer (unanswerable questions: the answer declines or states the correction).", "",
                _table(["Architecture", "Questions answered", "Accuracy, all questions (95% CI)", f"vs `{REFERENCE}`", "Answerable only", "Mean fact score", "Unanswerable handled", "Refused an answerable question"],
                       [[f"`{n}`", f"{len(stats[n]['all'])}/{meta['questions']}", _cell(stats[n]["all"], n in best),
                         "-" if n == REFERENCE or ref is None else _pp(_mean(stats[n]["all"]) - ref) if stats[n]["all"] else "-",
                         _cell(stats[n]["answerable"]), _pct(_mean(stats[n]["fact"])), _cell(stats[n]["unanswerable"]),
                         _pct(_mean(stats[n]["refusal"]))] for n in names]), ""]

    # ---- retrieval quality
    retrievers = [n for n in names if any(p > 0 for p in stats[n]["passages"])]     # not the no-retrieval baseline
    out += ["## Retrieval quality", "",
            "Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. "
            "These are measured on answerable questions only.", "",
            _table(["Architecture", "Context recall (95% CI)", "Page hit", "Gold pages read", "Pages read (mean)", "Context (characters)"],
                   [[f"`{n}`", _cell(stats[n]["ctx"], n in _best({m: _mean(stats[m]["ctx"]) for m in retrievers})), _cell(stats[n]["page_hit"]),
                     _pct(_mean(stats[n]["page_recall"])), _num(_mean(stats[n]["passages"])), _num(_mean(stats[n]["chars"]), "{:,.0f}")]
                    for n in retrievers]), ""]

    kinds = [t for t in TYPES if any(r["type"] == t and "context_recall" in r for r in grouped[names[0]])]
    sizes = [sum(1 for r in grouped[names[0]] if r["type"] == t and "context_recall" in r) for t in kinds]
    out += ["### Context recall by question type", "",
            _table(["Architecture"] + [f"{t} (n={s})" for t, s in zip(kinds, sizes)],
                   [[f"`{n}`"] + [_pct(_mean([r["context_recall"] for r in grouped[n] if r["type"] == t and "context_recall" in r])) for t in kinds]
                    for n in retrievers]), ""]

    # ---- cost
    if not retrieval_only:
        def per_call(n):
            calls = _mean(stats[n]["calls"])
            return "-" if not calls or not stats[n]["all"] else f"{100 * _mean(stats[n]['all']) / calls:.1f} pp"

        out += ["## Cost and speed", "",
                "Context tokens are the characters of the pages the LLM read divided by 4 (an estimate; the answer prompt and "
                "the extra calls of the longer pipelines add more). Seconds include the wait for the request-rate limit.", "",
                _table(["Architecture", "LLM calls per question", "Accuracy per LLM call", "Pages read (mean)", "Context tokens (estimate)",
                        "Median seconds", "95th percentile seconds", "Errors"],
                       [[f"`{n}`", _num(_mean(stats[n]["calls"])), per_call(n), _num(_mean(stats[n]["passages"])),
                         "-" if not stats[n]["chars"] else f"{_mean(stats[n]['chars']) / 4:,.0f}",
                         _num(_percentile(stats[n]["seconds"], 0.5)), _num(_percentile(stats[n]["seconds"], 0.95)), stats[n]["errors"]]
                        for n in names]), ""]

        # ---- self-verification
        checked = [n for n in names if any("supported" in r for r in grouped[n])]
        if checked:
            def verified_rows(n, flag):
                return [float(r["correct"]) for r in _answered(grouped[n]) if r.get("supported") is flag]

            out += ["## Self-verification (architectures with a confidence check)", "",
                    "Verified = the checker found the answer supported by the retrieved text. A useful check separates answers that are "
                    "right from answers that are wrong: accuracy should be higher among the verified ones.", "",
                    _table(["Architecture", "Mean confidence", "Answers verified", "Accuracy when verified", "Accuracy when not verified"],
                           [[f"`{n}`", _num(_mean([r.get("confidence") for r in grouped[n]]), "{:.2f}"),
                             _pct(_mean([float(r["supported"]) for r in grouped[n] if "supported" in r])),
                             _cell(verified_rows(n, True)), _cell(verified_rows(n, False))] for n in checked]), ""]

        routed = [n for n in names if any("route" in r for r in grouped[n])]
        for n in routed:
            share = {k: sum(1 for r in grouped[n] if r.get("route") == k) for k in "ABC"}
            out += [f"`{n}` routed {share['A']} questions to no retrieval, {share['B']} to single-step retrieval and {share['C']} to multi-step retrieval.", ""]

        # ---- by type and difficulty
        for title, key, groups in (("Accuracy by question type", "type", TYPES + ("false_premise", "unanswerable")),
                                   ("Accuracy by difficulty", "difficulty", ("easy", "medium", "hard"))):
            present = [g for g in groups if any(r[key] == g for r in grouped[names[0]])]
            table = []
            for n in names:
                done = _answered(grouped[n])
                table.append([f"`{n}`"] + [_pct(_mean([float(r["correct"]) for r in done if r[key] == g])) for g in present])
            sizes = [sum(1 for r in grouped[names[0]] if r[key] == g) for g in present]
            out += [f"## {title}", "", _table(["Architecture"] + [f"{g} (n={s})" for g, s in zip(present, sizes)], table), ""]

    # ---- paired comparison
    headline = HEADLINE if HEADLINE in grouped else "snaprag_fixed"

    if headline in grouped and len(names) > 1:
        head = {r["id"]: r for r in grouped[headline] if "error" not in r}
        rows = []
        for n in names:
            if n == headline:
                continue
            other = {r["id"]: r for r in grouped[n] if "error" not in r}
            acc = None if retrieval_only else _delta(head, other, "correct")
            fact = None if retrieval_only else _delta(head, other, "score")
            ctx = _delta(head, other, "context_recall")

            def show(d, test=False):
                if d is None:
                    return "-"
                mean, (low, high), wins, losses, size = d
                verdict = "better" if low > 0 else "worse" if high < 0 else "no clear difference"
                p = f"; sign test p={_sign_p(wins, losses):.3f}" if test else ""
                return f"{_pp(mean)} ({100 * low:+.0f} to {100 * high:+.0f}); {wins}/{losses}/{size - wins - losses}{p}; {verdict}"

            rows.append([f"`{n}`", "-" if retrieval_only else show(acc, True), "-" if retrieval_only else show(fact), show(ctx)])
        out += [f"## `{headline}` against every other architecture (paired, same questions)", "",
                "Each cell: mean difference in percentage points (95% bootstrap interval); questions won / lost / tied; "
                "'better' or 'worse' only when the interval excludes zero. The sign test is an exact two-sided binomial test "
                "on the questions where the two architectures differ.", "",
                _table(["Compared with", "Accuracy difference", "Fact-score difference", "Context-recall difference"], rows), ""]

    # ---- failures
    if HEADLINE in grouped and not retrieval_only:
        wrong = [r for r in _answered(grouped[HEADLINE]) if not r["correct"]]
        if wrong:
            out += [f"## Questions `{HEADLINE}` got wrong ({len(wrong)})", ""]
            for r in wrong:
                item = questions.get(r["id"], {})
                out.append(f"- **{r['id']}** ({r['type']}, {r['difficulty']}): {item.get('question', '')}  \n"
                           f"  expected: {item.get('answer', '')}  \n  answered: {' '.join(str(r.get('answer', '')).split())[:260]}")
            out.append("")

    out += ["## Notes", "",
            f"- One document, one LLM and {meta['questions']} questions written by reading the book: the intervals show how little this many questions can "
            "separate close systems. Treat differences smaller than the interval as noise.",
            "- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; "
            "they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.",
            "- The book is a scanned novel; some pages have broken text, which lowers every architecture equally.",
            "- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.",
            "- SnapRAG's AB-RAG weights and threshold are uncalibrated defaults.", ""]

    return "\n".join(out)


def write_report(result: dict, path: str) -> str:

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")
    dataset = json.load(open(dataset_path, encoding="utf-8")) if os.path.exists(dataset_path) else None

    with open(path, "w", encoding="utf-8") as f:
        f.write(render(result, dataset))

    return path


def _inline(text: str) -> str:
    """Markdown bold and code spans as the markup reportlab understands."""

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)

    return re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)


def write_pdf(markdown: str, path: str) -> str:
    """The report as a PDF (landscape A4): headings, tables, bullets and paragraphs."""

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle("body", parent=base, fontSize=9.5, leading=13, spaceAfter=5)
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10, spaceAfter=0)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    styles = {"# ": ParagraphStyle("h0", parent=body, fontName="Helvetica-Bold", fontSize=18, leading=22, spaceAfter=8),
              "## ": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10),
              "### ": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=6)}
    story, rows = [], []

    def flush():
        if not rows:
            return
        data = [[Paragraph(_inline(c.strip()), head if r == 0 else cell) for c in row] for r, row in enumerate(rows)]
        table = Table(data, repeatRows=1, colWidths=[(A4[1] - 3 * cm) / len(rows[0])] * len(rows[0]))
        table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                                   ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c2d0")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                   *[("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f1f4f8")) for r in range(2, len(rows), 2)]]))
        story.extend([table, Spacer(1, 6)])
        rows.clear()

    for line in markdown.split("\n"):

        if line.startswith("|"):
            cells = line.strip().strip("|").split("|")
            if not all(re.fullmatch(r"\s*-+\s*", c) for c in cells):
                rows.append(cells)
            continue

        flush()

        if line.strip() == "---":
            story.append(PageBreak())
        elif line.startswith(("### ", "## ", "# ")):
            prefix = next(p for p in ("### ", "## ", "# ") if line.startswith(p))
            story.append(Paragraph(_inline(line[len(prefix):]), styles[prefix]))
        elif line.startswith("- "):
            story.append(Paragraph(_inline(line[2:]), body, bulletText="•"))
        elif line.startswith("> "):
            story.append(Paragraph(_inline(line[2:]), body))
        elif line.strip():
            story.append(Paragraph(_inline(line), body))

    flush()
    SimpleDocTemplate(path, pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm,
                      bottomMargin=1.5 * cm, title="Document-QA architecture comparison").build(story)

    return path


def run_settings() -> dict:
    """The settings that shape the results, from the current configuration (for the reproducibility table)."""

    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(ROOT, "..", ".env"))
        from app.core.config import settings
    except Exception:
        return {}

    keys = ("LLM_MODEL", "LLM_REASONING_EFFORT", "EMBEDDING_MODEL", "CROSS_ENCODER_MODEL", "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE",
            "STOP_CONFIDENCE", "MAX_EXTRA_BATCHES", "CONF_W_MODEL", "CONF_W_SIMILARITY", "CONF_W_SPREAD")

    return {key: getattr(settings, key) for key in keys} | {"temperature": 0}


def to_latex(markdown: str) -> str:
    """Every Markdown table of the report as a booktabs table, headed by its section title."""

    def cell(text: str) -> str:
        for char in "&%_#$":
            text = text.replace(char, "\\" + char)

        return re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text.replace("`", ""))

    out, title, table = ["% generated by benchmarks.docqa.report", ""], "", []

    def flush():
        if not table:
            return
        width = len(table[0])
        out.extend(["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{title}}}", "\\begin{tabular}{l" + "r" * (width - 1) + "}", "\\toprule",
                    " & ".join(cell(c.strip()) for c in table[0]) + " \\\\", "\\midrule"]
                   + [" & ".join(cell(c.strip()) for c in row) + " \\\\" for row in table[1:]]
                   + ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
        table.clear()

    for line in markdown.split("\n"):

        if line.startswith("|"):
            cells = line.strip().strip("|").split("|")
            if not all(re.fullmatch(r"\s*-+\s*", c) for c in cells):
                table.append(cells)
            continue

        flush()

        if line.startswith("#"):
            title = line.lstrip("# ").replace("`", "").replace("_", "\\_").replace("%", "\\%")

    flush()

    return "\n".join(out)


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="results files; several are put one after the other in one report")
    parser.add_argument("--out", help="Markdown file (default: next to the first results file)")
    parser.add_argument("--pdf", help="also write the report as a PDF")
    parser.add_argument("--latex", help="also write every table as LaTeX (booktabs) for a paper")
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.json")
    dataset = json.load(open(dataset_path, encoding="utf-8")) if os.path.exists(dataset_path) else None
    results = [json.load(open(path, encoding="utf-8")) for path in args.results]

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

    text = "\n\n---\n\n".join(render(result, dataset) for result in results)
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
