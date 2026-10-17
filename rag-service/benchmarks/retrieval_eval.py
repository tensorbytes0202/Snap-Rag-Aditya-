"""Retrieval-only evaluation on the Chapin torts book: no answer is generated.

For every question it checks whether the context the LLM would read contains a
page from each section the question needs (ranges in legal_questions.json).

    python benchmarks/retrieval_eval.py            # real planner LLM call (1 per question)
    python benchmarks/retrieval_eval.py --no-llm   # every LLM call fails -> offline fallback path
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, "..", ".env"))

import logging

logging.disable(logging.CRITICAL)

from app.core.exceptions import LLMProviderError
from app.services import llm_provider

parser = argparse.ArgumentParser()
parser.add_argument("--no-llm", action="store_true")
args = parser.parse_args()

calls = {"n": 0}
_real_chat = llm_provider.chat


def counted_chat(*a, **k):
    calls["n"] += 1
    if args.no_llm:
        raise LLMProviderError("offline evaluation")
    return _real_chat(*a, **k)


# Every module imported `chat` by name, so patch them all.
from app.services.query_runner import run_retrieval

for module in list(sys.modules.values()):
    if getattr(module, "__name__", "").startswith("app.") and getattr(module, "chat", None) is _real_chat:
        module.chat = counted_chat

spec = json.load(open(os.path.join(ROOT, "benchmarks", "legal_questions.json"), encoding="utf-8"))
ranges = spec["sections"]

hit_total = need_total = 0

for item in spec["questions"]:
    calls["n"] = 0
    started = time.time()
    found = run_retrieval(item["question"], spec["document"])
    seconds = time.time() - started

    pages = sorted({chunk["page"] for chunk in found.top_chunks if chunk.get("page")})
    context_chars = len(found.top_context)

    hits = {
        need: any(ranges[need][0] <= page <= ranges[need][1] for page in pages)
        for need in item["needs"]
    }
    hit_total += sum(hits.values())
    need_total += len(hits)

    print(f"\n[{item['id']}]  {sum(hits.values())}/{len(hits)} sections covered  |  "
          f"{calls['n']} LLM calls  |  {seconds:.1f}s  |  {len(found.top_chunks)} chunks, {context_chars} chars")
    for need, ok in hits.items():
        print(f"   {'OK  ' if ok else 'MISS'} {need}")
    print("   pages in context:", pages)
    for query in found.rewritten_question:
        print("   query:", query)

print(f"\nTOTAL section coverage: {hit_total}/{need_total} = {hit_total / need_total:.0%}")
