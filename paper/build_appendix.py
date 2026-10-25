"""Writes paper/generated/appendix.tex and novelty.tex from the code, the benchmark spec and the notes.

    cd rag-service && ..\\venv\\Scripts\\python.exe ..\\paper\\build_appendix.py

The prompts, probes and hyper-parameters are read from the code itself, so the appendix cannot drift from
what was run.
"""

import json
import os
import re
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
RAG = os.path.join(os.path.dirname(HERE), "rag-service")
sys.path.insert(0, RAG)
os.chdir(RAG)

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))

from app.core.config import settings
from app.prompts.reconstruction_prompt import JUDGE_PROMPT, RECONSTRUCT_PROMPT
from app.prompts.snapshot_prompt import SNAPSHOT_PROMPT
from benchmarks.docqa import architectures

OUT = os.path.join(HERE, "generated")


def tex(text: str) -> str:
    """Plain text made safe for LaTeX."""

    for char in "&%_#$":
        text = text.replace(char, "\\" + char)

    return text


def listing(title: str, body: str) -> str:
    """A prompt in a verbatim block, wrapped so it stays inside the column."""

    body = body.replace("{{", "{").replace("}}", "}")        # the prompts are str.format templates
    lines = [chunk for line in body.strip().split("\n") for chunk in (textwrap.wrap(line, 88, subsequent_indent="    ") or [""])]

    return f"\\subsection{{{title}}}\n\\begin{{footnotesize}}\n\\begin{{verbatim}}\n" + "\n".join(lines) + "\n\\end{verbatim}\n\\end{footnotesize}\n"


def verification_prompt() -> str:
    """The answer-verification prompt lives inside the function; read it from the source."""

    source = open(os.path.join("app", "services", "verification_service.py"), encoding="utf-8").read()
    match = re.search(r'prompt = f"""(.*?)"""', source, re.S)

    return match.group(1).replace("{{", "{").replace("}}", "}") if match else "(see app/services/verification_service.py)"


def main() -> None:

    os.makedirs(OUT, exist_ok=True)
    spec = json.load(open(os.path.join("benchmarks", "longconv", "spec.json"), encoding="utf-8"))
    parts = ["\\section{Appendix}", "\\subsection*{Reproduction}",
             "\\begin{small}\\begin{verbatim}",
             "# the tests (no network, models or services needed)",
             "python -m pytest",
             "# long-conversation benchmark",
             "python -m benchmarks.longconv.run --variants <names> --turns 80 --repeats 3 --keep-chats benchmarks/results/chats",
             "python -m benchmarks.longconv.ladder --chats benchmarks/results/chats --variant full",
             "python -m benchmarks.verifier_audit sample benchmarks/results/ladder_full.json --out audit_sheet.json",
             "# document-QA architecture comparison",
             "python -m benchmarks.docqa.validate",
             "python -m benchmarks.docqa.run --workers 4",
             "python -m benchmarks.docqa.calibrate benchmarks/results/docqa_full75.json --test-from benchmarks/results/docqa_pilot30.json",
             "python -m benchmarks.docqa.report <results.json> --pdf report.pdf --latex tables.tex",
             "\\end{verbatim}\\end{small}"]

    parts += ["\\subsection*{Hyper-parameters}", "\\begin{table}[h]\\centering\\small\\begin{tabular}{ll}\\toprule Setting & Value \\\\ \\midrule"]
    keys = ("LLM_MODEL", "LLM_REASONING_EFFORT", "EMBEDDING_MODEL", "CROSS_ENCODER_MODEL", "PARENT_CHUNK_SIZE", "CHILD_CHUNK_SIZE",
            "MEMORY_BUDGET_TOKENS", "CQM_W_GROUNDING", "CQM_W_TOKEN", "CQM_W_COHERENCE", "CQM_W_CONTRADICTION", "CQM_TRIGGER_HEADROOM",
            "CQM_TRIGGER_CIS", "STOP_CONFIDENCE", "MAX_EXTRA_BATCHES", "CONF_W_MODEL", "CONF_W_SIMILARITY", "CONF_W_SPREAD", "HISTORY_TURNS")
    for key in keys:
        if hasattr(settings, key):
            parts.append(f"\\texttt{{{tex(key)}}} & {tex(str(getattr(settings, key)))} \\\\")
    parts += ["\\bottomrule\\end{tabular}\\caption{Settings of the runs (defaults of the code; the values the runs used are in each report's run-settings table).}\\end{table}"]

    parts += ["\\subsection*{Probes of \\bench{}}", "\\begin{small}\\begin{longtable}{p{0.07\\linewidth}p{0.12\\linewidth}p{0.75\\linewidth}}",
              "\\toprule Kind & Bucket & Statement and probe \\\\ \\midrule"]
    for plant in spec["plants"]:
        parts.append(f"{plant['kind']} & {plant['bucket']} & {tex(plant['statement'])} \\emph{{Probe:}} {tex(plant['probe'])} \\\\")
    for item in spec["corrections"]:
        parts.append(f"supersede & {item['bucket']} & {tex(item['statement'])} Later: {tex(item['correction'])} \\emph{{Probe:}} {tex(item['probe'])} \\\\")
    for item in spec["absent"]:
        parts.append(f"absent & -- & (never said) \\emph{{Probe:}} {tex(item['question'])} \\\\")
    for item in spec["traps"]:
        parts.append(f"trap & -- & (false premise) \\emph{{Probe:}} {tex(item['question'])} \\\\")
    parts.append(f"reference & -- & \\emph{{Probe (asked at four earlier turns):}} {tex(spec['references']['question'])} \\\\")
    parts += ["\\bottomrule\\end{longtable}\\end{small}"]

    parts += ["\\section{Prompts}",
              listing("Snapshot encoder", SNAPSHOT_PROMPT),
              listing("Reconstruction", RECONSTRUCT_PROMPT),
              listing("Sentence judge (verification)", JUDGE_PROMPT),
              listing("Answer verification (confidence check)", verification_prompt())]

    for name in ("GRADE_PROMPT", "CRITIC_PROMPT", "ROUTE_PROMPT", "STEP_PROMPT", "REWRITE_PROMPT"):
        parts.append(listing(f"Baseline prompt: {name.replace('_PROMPT', '').lower()} (re-implemented method)", getattr(architectures, name)))

    with open(os.path.join(OUT, "appendix.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    novelty = os.path.join(HERE, "novelty_check.tex")
    print("appendix written;", "novelty notes present" if os.path.exists(novelty) else "write paper/novelty_check.tex")


if __name__ == "__main__":
    main()
