"""Composes the project's results file from whatever result files exist.

    python -m benchmarks.assemble_results

Writes SnapRAG_Benchmark_Results.md / .pdf and SnapRAG_Benchmark_Tables.tex in the project root: the status of the plan
(paper/status.md), the long-conversation results (with the verification ladder and the verifier audit), the document-QA
comparison, the calibration, the retrieval-only comparison on all 75 questions, the earlier 30-question pilot and the
novelty check.
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT = os.path.dirname(ROOT)
RESULTS = os.path.join(ROOT, "benchmarks", "results")
sys.path.insert(0, ROOT)

from benchmarks.docqa import report as docqa_report
from benchmarks.docqa.export import to_latex, write_pdf
from benchmarks.longconv import final_report
from benchmarks.memqa import report as memqa_report


def read(path: str) -> str:

    return open(path, encoding="utf-8").read() if os.path.exists(path) else ""


def docqa_section(path: str, dataset) -> str:

    if not os.path.exists(path):
        return ""

    result = json.load(open(path, encoding="utf-8"))
    docqa_report.prepare([result])

    return docqa_report.render(result, dataset)


def main() -> None:

    dataset = docqa_report._dataset()
    sections = [read(os.path.join(PROJECT, "paper", "status.md"))]

    parts = sorted(glob.glob(os.path.join(RESULTS, "longconv_part*.json")))

    if parts:
        by_variant, meta = final_report.pool(parts)
        ladder = os.path.join(RESULTS, "ladder_full.json")
        audit = os.path.join(RESULTS, "audit_sheet.json")
        ladder_data = json.load(open(ladder, encoding="utf-8")) if os.path.exists(ladder) else None
        audit_data = None

        if os.path.exists(audit):
            from benchmarks.verifier_audit import score
            audit_data = score(json.load(open(audit, encoding="utf-8")))

        sections.append(final_report.render(by_variant, meta, ladder_data, audit_data))

    for name in ("docqa_full75.json", "docqa_second_llm.json"):
        sections.append(docqa_section(os.path.join(RESULTS, name), dataset))

    memqa = [p for p in sorted(glob.glob(os.path.join(RESULTS, "memqa_*.json"))) if "smoke" not in p]

    if memqa:
        intro = ("## Conversational-memory QA (LoCoMo, LongMemEval)\n\nSampled questions, lexical scoring (token F1 >= 0.5 or the gold answer contained); "
                 "LongMemEval is the oracle file (evidence sessions only). A small sample: read the intervals, not the point estimates.\n\n")
        sections.append(intro + "\n\n".join(memqa_report.render(p) for p in memqa))

    sections.append(read(os.path.join(RESULTS, "docqa_calibration.md")))
    sections.append(read(os.path.join(RESULTS, "docqa_ablations.md")))
    sections.append(docqa_section(os.path.join(RESULTS, "docqa_retrieval.json"), dataset))
    sections.append(docqa_section(os.path.join(RESULTS, "docqa_pilot30.json"), dataset).replace(
        "# Document-QA architecture comparison -- answers", "# Pilot: the first 30-question comparison (before token counting, calibration and the full run)", 1))
    sections.append(read(os.path.join(PROJECT, "paper", "novelty_check.md")))

    text = "\n\n---\n\n".join(section for section in sections if section.strip())

    with open(os.path.join(PROJECT, "SnapRAG_Benchmark_Results.md"), "w", encoding="utf-8") as f:
        f.write(text)

    write_pdf(text, os.path.join(PROJECT, "SnapRAG_Benchmark_Results.pdf"))

    with open(os.path.join(PROJECT, "SnapRAG_Benchmark_Tables.tex"), "w", encoding="utf-8") as f:
        f.write(to_latex(text))

    print("written SnapRAG_Benchmark_Results.md / .pdf and SnapRAG_Benchmark_Tables.tex")


if __name__ == "__main__":
    main()
