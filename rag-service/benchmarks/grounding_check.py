"""Answer a question from legal_questions.json end to end, then check that
every case name and every "(PDF p. N)" citation in the answer really occurs in
the context the LLM was given. Costs about 4 LLM calls per question.

    python benchmarks/grounding_check.py railway-natural-event fright-mental-anguish
"""

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, "..", ".env"))

import logging

logging.disable(logging.CRITICAL)

from app.services import answering, query_service

seen = {}
_generate = answering.generate_answer


def capture(question, context, *args, **kwargs):
    seen["context"] = context
    return _generate(question, context, *args, **kwargs)


answering.generate_answer = capture
query_service.get_cache = lambda key: None  # always answer afresh
query_service.set_cache = lambda key, value: None

spec = json.load(open(os.path.join(ROOT, "benchmarks", "legal_questions.json"), encoding="utf-8"))
wanted = sys.argv[1:] or [item["id"] for item in spec["questions"]]

_NAME = r"[A-Z][\w.'&-]*"
CASE = re.compile(rf"({_NAME}(?: {_NAME}){{0,3}}) v\. ({_NAME}(?: {_NAME}){{0,3}})")


def squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


for item in spec["questions"]:

    if item["id"] not in wanted:
        continue

    started = time.time()
    response = query_service.process_query(item["question"], spec["document"])
    seconds = time.time() - started

    answer = response["answer"]
    context = squash(seen.get("context", ""))
    pages = {int(n) for n in re.findall(r"PDF page (\d+)\]", seen.get("context", ""))}

    cases = sorted({m.group(0) for m in CASE.finditer(answer)})
    # OCR'd names vary ("Adm'r", "V." or "y." for "v."), so only the first word of each party is matched.
    grounded = [
        c for c in cases
        if re.search(
            rf"{re.escape(c.split(' v. ')[0].split()[0])}\W+(?:[\w.&' ]{{0,60}}?)[vy]\.?\s+{re.escape(c.split(' v. ')[1].split()[0])}",
            context,
            re.IGNORECASE,
        )
    ]
    cited = {int(n) for n in re.findall(r"PDF pp?\.? ?(\d+)", answer)}

    print("=" * 78)
    print(f"[{item['id']}]  {seconds:.1f}s  |  verified: {response['verification'].get('supported')}  |  confidence {response['confidence']}")
    print(f"cases named: {len(cases)}, found in context: {len(grounded)}  ->  not in context: {sorted(set(cases) - set(grounded))}")
    print(f"pages cited: {sorted(cited)}  ->  not in context: {sorted(cited - pages)}")
    print("-" * 78)
    print(answer)
