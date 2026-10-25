"""The verification ladder on a long conversation: unsupported-claim rate and fact recall.

run.py --keep-chats DIR keeps the finished conversation of each variant. This rebuilds the
memory for the benchmark's memory probes in each mode (snapshot -> evidence -> verify -> archive)
and has a separate judge model score what each mode would put in a prompt:

    unsupported-claim rate  returned sentences the archive and the anchored pages do not support / returned
    fact recall             reference facts (taken from the archive) that the returned memory states / reference facts

    python -m benchmarks.longconv.ladder --chats benchmarks/results/chats --variant full --turns 80 --judge-model gemini-3-flash-preview
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, "..", ".env"))

import logging

logging.disable(logging.CRITICAL)

from app.core.config import settings
from benchmarks import reconstruction_eval
from benchmarks.longconv.run import USER
from benchmarks.longconv.script import build_script, load_spec

MEMORY_KINDS = ("recall", "supersede", "reference")


def probe_questions(turns: int, seed: int) -> list[str]:
    """The questions of the memory probes of the script (what the ladder is asked to rebuild)."""

    script = build_script(load_spec(), turns, seed)

    return [item["text"] for item in script if item["kind"] == "probe" and item["probe_kind"] in MEMORY_KINDS]


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chats", required=True, help="the --keep-chats folder of the run")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--turns", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--judge-model", default="gemini-3-flash-preview", help="scoring model; keep it different from the reconstruction model")
    parser.add_argument("--out")
    args = parser.parse_args()

    settings.MEMORY_ROOT = os.path.join(args.chats, f"{args.variant}_r{args.repeat}")

    from app.services.session_store import list_chats

    chats = list_chats(USER)

    if not chats:
        sys.exit(f"no conversation under {settings.MEMORY_ROOT}")

    chat_id = chats[0]["chat_id"] if isinstance(chats[0], dict) else chats[0]
    probes = probe_questions(args.turns, args.seed)
    print(f"{len(probes)} memory probes on {args.variant} (chat {chat_id}); judge {args.judge_model}", flush=True)

    report = reconstruction_eval.evaluate(USER, chat_id, probes, args.judge_model)
    report.update(variant=args.variant, turns=args.turns, seed=args.seed, date=time.strftime("%Y-%m-%d %H:%M"))

    out = args.out or os.path.join(ROOT, "benchmarks", "results", f"ladder_{args.variant}.json")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print(f"\nLadder results: {out}")

    for mode, s in report["summary"].items():
        print(f"  {mode:9s} unsupported {s['hallucination_rate']:.1%}  fact recall {s['fact_recall']}  returned {s['sentences_returned']}  dropped {s['sentences_dropped']}")


if __name__ == "__main__":
    main()
