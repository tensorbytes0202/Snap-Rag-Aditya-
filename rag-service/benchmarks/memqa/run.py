"""Conversational-memory QA on LoCoMo and LongMemEval (no document: the answer is in the chat history).

    ..\\venv\\Scripts\\python.exe -m benchmarks.memqa.run --dataset locomo --systems no_memory,full_history,snapshot,full --questions 30
    ..\\venv\\Scripts\\python.exe -m benchmarks.memqa.run --dataset longmemeval --systems no_memory,full_history,full --questions 20

The recorded dialogue is loaded into a chat as ready-made turns (no LLM answers it), snapshots are cut over it the way
production does (`after_turn`, every `--upkeep-every` turns to save calls), then the questions are asked. Systems are the
variants of `benchmarks/longconv/variants.py`. LongMemEval uses the *oracle* file (only the evidence sessions), so it tests
recall and update over a few sessions, not the 115k-token haystack of the S setting; this is stated in the report.
Scoring is lexical (token F1 and "gold answer contained"); abstention questions are scored by the refusal pattern.
"""

import argparse
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, "..", ".env"))

from benchmarks.atomic import dump_json  # noqa: E402
from benchmarks.docqa.scoring import DECLINE  # noqa: E402
from benchmarks.longconv import variants as variant_defs  # noqa: E402
from benchmarks.longconv.run import Conversation, answers_not_cached, git_commit, isolated_storage, overridden  # noqa: E402

USER = "benchmark-user"
DATA = os.path.join(ROOT, "benchmarks", "data")
OUT = os.path.join(ROOT, "benchmarks", "results")
LOCOMO_CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial (unanswerable)"}


# ==========================================================
# Datasets -> (turns, questions)
# ==========================================================

def load_locomo(count: int, per_conversation: int, seed: int) -> list[dict]:

    data = json.load(open(os.path.join(DATA, "locomo10.json"), encoding="utf-8"))[:count]
    rng = random.Random(seed)
    items = []

    for sample in data:

        conversation, lines = sample["conversation"], []

        for n in range(1, 60):
            session = conversation.get(f"session_{n}")
            if not session:
                continue
            date = conversation.get(f"session_{n}_date_time", "")
            lines += [f"[{date}] {u['speaker']}: {u['text']}" for u in session]

        turns = [{"question": lines[i], "answer": lines[i + 1] if i + 1 < len(lines) else ""} for i in range(0, len(lines), 2)]
        questions = []

        for q in sample["qa"]:
            abstain = q["category"] == 5
            questions.append({"question": q["question"], "gold": None if abstain else str(q["answer"]),
                              "abstain": abstain, "type": LOCOMO_CATEGORY[q["category"]], "evidence": q.get("evidence", [])})

        # An even share of every category, so the small sample still covers all five.
        by_type = {}
        for q in questions:
            by_type.setdefault(q["type"], []).append(q)
        chosen = []
        for group in by_type.values():
            rng.shuffle(group)
            chosen += group[:max(1, per_conversation // len(by_type))]

        items.append({"id": sample["sample_id"], "turns": turns, "questions": chosen})

    return items


def load_longmemeval(count: int, seed: int) -> list[dict]:

    data = json.load(open(os.path.join(DATA, "longmemeval_oracle.json"), encoding="utf-8"))
    rng = random.Random(seed)
    by_type = {}

    for q in data:
        by_type.setdefault(q["question_type"] + ("/abstention" if q["question_id"].endswith("_abs") else ""), []).append(q)

    chosen = []
    for group in by_type.values():
        rng.shuffle(group)
        chosen += group[:max(1, count // len(by_type))]

    items = []

    for q in chosen:
        turns = []
        for date, session in zip(q["haystack_dates"], q["haystack_sessions"]):
            for i in range(0, len(session), 2):
                answer = session[i + 1]["content"] if i + 1 < len(session) else ""
                turns.append({"question": f"[{date}] {session[i]['content']}", "answer": answer})
        abstain = q["question_id"].endswith("_abs")
        items.append({"id": q["question_id"], "turns": turns, "questions": [{
            "question": f"(Today is {q['question_date']}.) {q['question']}", "gold": None if abstain else str(q["answer"]),
            "abstain": abstain, "type": q["question_type"] + ("/abstention" if abstain else ""), "evidence": q["answer_session_ids"]}]})

    return items


# ==========================================================
# Scoring
# ==========================================================

def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).split()


def f1(answer: str, gold: str) -> float:

    a, g = _norm(answer), _norm(gold)
    common = sum(min(a.count(w), g.count(w)) for w in set(g))

    if not common:
        return 0.0

    precision, recall = common / len(a), common / len(g)

    return 2 * precision * recall / (precision + recall)


def score(question: dict, answer: str) -> dict:

    declined = re.search(DECLINE, answer, re.I) is not None

    if question["abstain"]:
        return {"correct": declined, "f1": None, "declined": declined}

    gold = question["gold"]
    contained = " ".join(_norm(gold)) in " ".join(_norm(answer))
    value = f1(answer, gold)

    return {"correct": (contained or value >= 0.5) and not declined, "f1": round(value, 3), "declined": declined}


# ==========================================================
# Running
# ==========================================================

def run_system(name: str, item: dict, *, upkeep_every: int, log=print) -> dict:

    from app.core.metrics import count_llm_calls
    from app.services.query_service import process_query
    from app.services.session_store import append_turn, create_chat

    variant = variant_defs.VARIANTS[name]

    with overridden({**variant.settings, "FOLLOWUPS_ENABLED": False}), isolated_storage(), answers_not_cached():

        chat = create_chat(USER)
        conversation = Conversation(variant, USER, chat)

        with count_llm_calls() as build:
            for n, turn in enumerate(item["turns"], 1):
                append_turn(USER, chat, turn["question"], turn["answer"])
                if variant.uses_snapshots and n % upkeep_every == 0:
                    conversation.upkeep()
            if variant.uses_snapshots:
                conversation.upkeep()

        stats = conversation.snapshot_stats()
        rows = []

        for q in item["questions"]:

            history = conversation.history()

            with count_llm_calls() as used:
                started = time.perf_counter()
                try:
                    response, error = process_query(q["question"], None, history, conversation.session()), None
                except Exception as exc:
                    response, error = None, str(exc)[:200]
                seconds = time.perf_counter() - started

            row = {"id": item["id"], "system": name, "type": q["type"], "question": q["question"], "gold": q["gold"],
                   "abstain": q["abstain"], "error": error, "calls": used["n"], "tokens_in": used["prompt_tokens"],
                   "tokens_out": used["completion_tokens"], "active_seconds": round(max(0.0, seconds - used["wait"]), 2)}

            if response is not None:
                row.update(answer=response["answer"], **score(q, response["answer"]))

            rows.append(row)
            log(f"  [{name}] {item['id']} {q['type'][:22]:22s} {'ERR' if error else ('OK ' if row.get('correct') else 'no ')} {seconds:5.1f}s")

    return {"system": name, "id": item["id"], "build_calls": build["n"], "build_tokens_in": build["prompt_tokens"],
            "build_tokens_out": build["completion_tokens"], "snapshots": stats["snapshots"], "state_items": stats["state_items"], "rows": rows}


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--systems", default="no_memory,full_history,snapshot,full")
    parser.add_argument("--conversations", type=int, default=1, help="LoCoMo conversations")
    parser.add_argument("--questions", type=int, default=30, help="questions per conversation (LoCoMo) or in total (LongMemEval)")
    parser.add_argument("--upkeep-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--out")
    args = parser.parse_args()

    items = load_locomo(args.conversations, args.questions, args.seed) if args.dataset == "locomo" else load_longmemeval(args.questions, args.seed)
    out = args.out or os.path.join(OUT, f"memqa_{args.dataset}.json")
    runs = json.load(open(args.resume, encoding="utf-8"))["runs"] if args.resume and os.path.exists(args.resume) else []
    done = {(r["system"], r["id"]) for r in runs}

    from app.core.config import settings

    for item in items:
        for name in args.systems.split(","):

            if (name, item["id"]) in done:
                continue

            print(f"\n== {args.dataset} {item['id']}: {name} ({len(item['turns'])} turns, {len(item['questions'])} questions)", flush=True)
            runs.append(run_system(name, item, upkeep_every=args.upkeep_every))

            dump_json({"meta": {"dataset": args.dataset, "systems": args.systems, "seed": args.seed, "model": settings.LLM_MODEL,
                                "date": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit(), "upkeep_every": args.upkeep_every},
                       "runs": runs}, out)

    print("saved", out)


if __name__ == "__main__":
    main()
