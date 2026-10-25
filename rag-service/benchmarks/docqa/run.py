"""Runs the document-QA architecture comparison and writes a results file and a report.

    python -m benchmarks.docqa.run                         # every architecture, every question
    python -m benchmarks.docqa.run --archs dense_k5,snaprag --limit 10
    python -m benchmarks.docqa.run --retrieval-only        # no LLM: retrieval metrics only
    python -m benchmarks.docqa.run --resume benchmarks/results/docqa_<time>.json

Needs Qdrant with the document indexed (upload it once). Results are saved after
every answer, so an interrupted run continues with --resume.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, "..", ".env"))
load_dotenv(os.path.join(ROOT, ".env"))

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.metrics import count_llm_calls
from app.services import llm_provider
from benchmarks.atomic import dump_json
from benchmarks.docqa import architectures
from benchmarks.docqa.report import write_report
from benchmarks.docqa.scoring import score_answer, score_retrieval


def offline_llm() -> None:
    """Makes every LLM call fail, so only the retrieval (with SnapRAG's offline
    fallbacks for the planner and HyDE) is measured."""

    real = llm_provider.chat

    def refuse(*args, **kwargs):
        raise LLMProviderError("retrieval-only run")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("app.") and getattr(module, "chat", None) is real:
            module.chat = refuse


def _commit() -> str:
    """The short git commit the run was made from (with a * when the tree has uncommitted changes)."""

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        return head + ("*" if dirty else "")
    except Exception:
        return "unknown"


def ask(arch: str, item: dict, document: str, offline: bool) -> dict:
    """One architecture answering one question; a row of the results file."""

    function = architectures.ARCHITECTURES[arch][1]
    row = {"arch": arch, "id": item["id"], "type": item["type"], "difficulty": item["difficulty"],
           "answerable": item.get("answerable", True)}
    started = time.time()

    try:
        with count_llm_calls() as calls:
            out = function(item["question"], document, offline=offline)
    except Exception as exc:
        return {**row, "error": f"{type(exc).__name__}: {exc}"[:300]}

    pages = sorted({chunk["page"] for chunk in out["chunks"] if chunk.get("page")})
    seconds = time.time() - started
    row.update(seconds=round(seconds, 2), active_seconds=round(max(0.0, seconds - calls["wait"]), 2), calls=calls["n"],
               tokens_in=calls["prompt_tokens"], tokens_out=calls["completion_tokens"], pages=pages, passages=len(out["chunks"]),
               context_chars=len(out["context"]), answer=out["answer"])

    for key in ("confidence", "supported", "route", "attempts"):
        if key in out:
            row[key] = out[key]

    if out["answer"] is not None:
        row.update(score_answer(item, out["answer"]))

    row.update(score_retrieval(item, out["context"], pages) or {})

    return row


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archs", help="comma-separated names (default: all)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--sample", type=int, help="an evenly spaced sample of N questions, the same for every architecture")
    parser.add_argument("--workers", type=int, default=1, help="questions answered in parallel")
    parser.add_argument("--retrieval-only", action="store_true", help="no LLM calls: retrieval metrics only")
    parser.add_argument("--resume", help="results file of an interrupted run")
    parser.add_argument("--out", help="results file (default benchmarks/results/docqa_<time>.json)")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)

    data = json.load(open(os.path.join(ROOT, "benchmarks", "docqa", "dataset.json"), encoding="utf-8"))
    questions = data["questions"][:args.limit]

    if args.sample:
        questions = [questions[int(i * len(questions) / args.sample)] for i in range(args.sample)]
    names = args.archs.split(",") if args.archs else (architectures.RETRIEVAL_ONLY if args.retrieval_only else list(architectures.ARCHITECTURES))

    for name in names:
        if name not in architectures.ARCHITECTURES:
            sys.exit(f"unknown architecture {name!r}; choose from {', '.join(architectures.ARCHITECTURES)}")

    path = args.resume or args.out or os.path.join(ROOT, "benchmarks", "results", f"docqa_{time.strftime('%Y%m%d_%H%M%S')}.json")
    saved = json.load(open(path, encoding="utf-8")) if args.resume and os.path.exists(path) else None
    result = saved or {"meta": {"document": data["document"], "model": settings.LLM_MODEL, "started": time.strftime("%Y-%m-%d %H:%M"),
                                "retrieval_only": args.retrieval_only, "questions": len(questions), "commit": _commit(), "seed": 0,
                                "temperature": 0, "sample": args.sample}, "rows": []}
    result["meta"]["ids"] = [item["id"] for item in questions]
    result["meta"]["questions"] = len(questions)
    result["meta"]["archs"] ={**result["meta"].get("archs", {}), **{name: architectures.ARCHITECTURES[name][0] for name in names}}
    result["meta"]["sources"] = {**result["meta"].get("sources", {}), **{name: architectures.ARCHITECTURES[name][3] for name in names}}
    done = {(row["arch"], row["id"]) for row in result["rows"] if "error" not in row}
    result["rows"] = [row for row in result["rows"] if (row["arch"], row["id"]) in done]

    if args.retrieval_only:
        offline_llm()

    # Warm the models once so the first question's time is not the load time.
    from app.api.dependencies.embedding import get_embedding_model
    from app.api.dependencies.reranker import get_cross_encoder
    get_embedding_model(), get_cross_encoder()

    from app.services import query_service
    query_service.get_cache = lambda key: None
    query_service.set_cache = lambda key, value: None

    def save() -> None:
        dump_json(result, path)

    for name in names:

        todo = [item for item in questions if (name, item["id"]) not in done]
        print(f"== {name}: {architectures.ARCHITECTURES[name][0]} ({len(todo)} questions)", flush=True)
        previous = architectures.apply_settings(architectures.ARCHITECTURES[name][2])

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for item, row in zip(todo, pool.map(lambda item: ask(name, item, data["document"], args.retrieval_only), todo)):
                    result["rows"].append(row)
                    save()
                    flag = "ERROR " + row["error"][:80] if "error" in row else ("OK  " if row.get("correct") else "    ") if "correct" in row else ""
                    print(f"  {item['id']} {item['type']:<13} {row.get('seconds', 0):>6.1f}s  calls {row.get('calls', 0):>2}  {flag}", flush=True)
        finally:
            architectures.apply_settings(previous)

    result["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M")
    save()
    report = write_report(result, os.path.splitext(path)[0] + ".md")
    print(f"\nResults: {path}\nReport:  {report}")


if __name__ == "__main__":
    main()
