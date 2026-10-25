"""Reconstruction benchmark: the Phase 2b ablation ladder.

For each probe question about turns a snapshot replaced, every mode
(snapshot -> evidence -> verify -> archive) rebuilds the memory. A separate,
stronger judge model then scores what each mode would put into a prompt:

    hallucination  returned sentences that neither the archive nor the
                   snapshot's anchored pages support / returned sentences
    fact recall    reference facts for the probe (extracted once from the
                   archive) that the returned memory states / reference facts
    memory tokens  size of the returned memory
    compression    the snapshot's source tokens / snapshot tokens

Run from rag-service/:
    ..\\venv\\Scripts\\python.exe -m benchmarks.reconstruction_eval --demo
    ..\\venv\\Scripts\\python.exe -m benchmarks.reconstruction_eval --user dev-user --chat <id> --question "..."
"""

import argparse
import json
import os
import sys
import tempfile
import time

from dotenv import load_dotenv

load_dotenv()

from app.core.config import settings  # noqa: E402  (after load_dotenv)

DEMO_DOCUMENT = "implementation of static routing using cisco packet tracer.pdf"

DEMO_TURNS = [
    "What static routes are configured on Router-0?",
    "What about Router-2?",
    "What IP address and default gateway does PC5 use?",
    "For my report I will only describe the Router-0 and Router-2 static routes and skip Router-1. "
    "Which pages show those two routers' static routes?",
    "Which links connect Router-1 to Router-0 and Router-2?",
]

DEMO_PROBES = [
    "What static routes are configured on Router-0 and Router-2?",
    "What default gateway does PC5 use?",
    "Which routers did I say my report will describe?",
    "Which serial interfaces connect Router-1 to the other routers?",
    "What OSPF area is configured on Router-1?",  # nothing in memory: should come back empty
]

REFERENCE_PROMPT = """CONVERSATION:
{archive}

QUESTION: {question}

List every atomic fact from this conversation (assistant answers or statements the user made)
that is needed to answer the QUESTION. Keep numbers and names exact. Empty list if none.
Return ONLY JSON: {{"facts": ["...", "..."]}}"""


def with_retry(call, attempts: int = 3, wait_seconds: int = 30):
    """Run `call` until it returns something other than None. Quota limits
    (a 429 surfaces as an LLM error) are per minute, so wait between tries.
    Returns None if every attempt failed."""

    from app.core.exceptions import LLMProviderError

    for attempt in range(attempts):

        try:
            result = call()
        except (LLMProviderError, ValueError):
            result = None

        if result is not None:
            return result

        if attempt < attempts - 1:
            time.sleep(wait_seconds)

    return None


def _tokens(text: str) -> int:

    from app.services.context_monitor import CHARS_PER_TOKEN

    return len(text) // CHARS_PER_TOKEN


def build_demo_chat() -> tuple[str, str]:
    """A fresh 5-turn chat on the demo PDF in temporary storage, compressed
    into one snapshot."""

    from app.services import session_store
    from app.services.query_service import process_query
    from app.services.snapshot_engine import create_snapshot

    user, chat_id = "benchmark", session_store.create_chat("benchmark")

    for question in DEMO_TURNS:
        history = session_store.get_turns(user, chat_id)[-settings.HISTORY_TURNS:]
        response = with_retry(lambda: process_query(question, DEMO_DOCUMENT, history))
        if response is None:
            sys.exit("LLM unavailable while building the demo chat; try again later.")
        verification = response.get("verification")
        session_store.append_turn(
            user, chat_id, question, response["answer"],
            document=DEMO_DOCUMENT, citations=response.get("citations", []),
            verified=verification.get("supported", False) if isinstance(verification, dict) else False,
        )
        print(f"  turn built: {question[:60]}", flush=True)

    # The trigger is not under test here, so the snapshot is taken directly.
    turns = session_store.get_turns(user, chat_id)
    if with_retry(lambda: create_snapshot(user, chat_id, turns, [], {"trigger": "benchmark"})) is None:
        sys.exit("LLM unavailable while creating the snapshot; try again later.")

    return user, chat_id


def best_snapshot(snapshots: list[dict], question: str, turns: list[dict] | None = None) -> dict:
    """The snapshot whose span of the conversation fits the question best. With `turns` it is matched on the original
    turns (the lossy summary can miss the very fact the probe asks for); without, on the rendered snapshot."""

    from app.api.dependencies.embedding import get_embedding_model
    from app.services.snapshot_engine import render_snapshot

    if len(snapshots) == 1:
        return snapshots[0]

    model = get_embedding_model()
    q = model.encode(question, normalize_embeddings=True)

    def text(s):
        if turns is None:
            return render_snapshot(s)
        return " ".join(t["question"] + " " + t["answer"] for t in turns if s["from_turn"] <= t["turn"] <= s["to_turn"]) or render_snapshot(s)

    return max(snapshots, key=lambda s: float(q @ model.encode(text(s), normalize_embeddings=True)))


def evaluate(user: str, chat_id: str, probes: list[str], judge_model: str) -> dict:

    from app.prompts.reconstruction_prompt import render_archive, render_evidence
    from app.services.llm_provider import chat
    from app.services.reconstruction import MODES, judge, load_evidence, recall
    from app.services.session_store import get_turns
    from app.services.snapshot_store import list_snapshots
    from app.utils.json_utils import extract_json

    snapshots = list_snapshots(user, chat_id)

    if not snapshots:
        sys.exit("This chat has no snapshot yet; nothing to reconstruct.")

    rows = []

    for question in probes:

        snapshot = best_snapshot(snapshots, question, get_turns(user, chat_id))
        archive = [t for t in get_turns(user, chat_id) if snapshot["from_turn"] <= t["turn"] <= snapshot["to_turn"]]
        truth = render_archive(archive) + "\n\n" + render_evidence(load_evidence(snapshot))

        def reference_facts():
            reply = chat([{"role": "user", "content": REFERENCE_PROMPT.format(
                archive=render_archive(archive), question=question)}], model=judge_model)
            data = extract_json(reply["message"]["content"])
            facts = data.get("facts") if isinstance(data, dict) else None
            return [f.strip() for f in facts if isinstance(f, str) and f.strip()] if isinstance(facts, list) else None

        reference = with_retry(reference_facts)

        print(f"\nPROBE: {question}  (snapshot {snapshot['seq']}, "
              f"{'?' if reference is None else len(reference)} reference facts)", flush=True)

        for mode in MODES:

            def verified_recall():
                result = recall(user, chat_id, snapshot, question, mode)
                # A failed check is an outage, not the method's answer: retry it.
                return None if result["verification_failed"] else result

            result = with_retry(verified_recall)

            if result is None:
                rows.append({"question": question, "mode": mode, "error": "LLM unavailable"})
                print(f"  {mode:9s} ERROR: LLM unavailable after retries", flush=True)
                continue

            sentences = [item["text"] for item in result["kept"]]
            memory = "\n".join(sentences)

            # None = the judge could not score it; such rows are left out of
            # the averages instead of being counted as 0.
            truth_labels = with_retry(lambda: judge(sentences, {"truth": truth}, model=judge_model))
            recall_labels = (
                with_retry(lambda: judge(reference, {"memory": memory or "(empty)"}, model=judge_model))
                if reference else None
            )

            unsupported = None if truth_labels is None else [
                s for s, label in zip(sentences, truth_labels) if label != "truth"
            ]

            row = {
                "question": question,
                "mode": mode,
                "returned": len(sentences),
                "dropped": len(result["dropped"]),
                "unsupported": None if unsupported is None else len(unsupported),
                "fact_recall": round(recall_labels.count("memory") / len(reference), 3) if recall_labels is not None else None,
                "memory_tokens": _tokens(memory),
                "kept": result["kept"],
                "dropped_sentences": result["dropped"],
                "unsupported_sentences": unsupported,
                "sources": truth,       # what a sentence may be supported by: the archive and the pages the snapshot cites
            }
            rows.append(row)

            print(f"  {mode:9s} returned={row['returned']:2d} dropped={row['dropped']:2d} "
                  f"unsupported={row['unsupported']} recall={row['fact_recall']} tokens={row['memory_tokens']}", flush=True)

    summary = {}

    for mode in MODES:
        mine = [r for r in rows if r["mode"] == mode and "error" not in r]
        judged = [r for r in mine if r["unsupported"] is not None]
        returned = sum(r["returned"] for r in judged)
        recalls = [r["fact_recall"] for r in mine if r["fact_recall"] is not None]
        summary[mode] = {
            "hallucination_rate": round(sum(r["unsupported"] for r in judged) / returned, 3) if returned else 0.0,
            "fact_recall": round(sum(recalls) / len(recalls), 3) if recalls else None,
            "avg_memory_tokens": round(sum(r["memory_tokens"] for r in mine) / len(mine), 1) if mine else None,
            "sentences_returned": sum(r["returned"] for r in mine),
            "sentences_dropped": sum(r["dropped"] for r in mine),
            "probes_scored": len(mine),
            "errors": sum(1 for r in rows if r["mode"] == mode and "error" in r),
        }

    return {
        "chat_id": chat_id,
        "judge_model": judge_model,
        "reconstruction_model": settings.RECONSTRUCTION_MODEL or settings.LLM_MODEL,
        "compression": [s["meta"].get("compression") for s in snapshots],
        "summary": summary,
        "rows": rows,
    }


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", action="store_true", help="build a 5-turn demo chat in temporary storage")
    parser.add_argument("--user", help="user id that owns --chat")
    parser.add_argument("--chat", help="chat id with at least one snapshot")
    parser.add_argument("--question", action="append", help="probe question (repeatable)")
    parser.add_argument("--judge-model", default="gemini-3.5-flash",
                        help="scoring model; keep it different from the reconstruction model")
    args = parser.parse_args()

    if args.demo:
        # Never touch real chats or the real snapshot index.
        settings.MEMORY_ROOT = tempfile.mkdtemp(prefix="snaprag_bench_")
        settings.SNAPSHOT_COLLECTION = "snapshot_memory_benchmark"
        print("Building demo chat...", flush=True)
        user, chat_id = build_demo_chat()
        probes = args.question or DEMO_PROBES
    elif args.user and args.chat and args.question:
        user, chat_id, probes = args.user, args.chat, args.question
    else:
        parser.error("use --demo, or --user, --chat and at least one --question")

    report = evaluate(user, chat_id, probes, args.judge_model)

    print(f"\nSUMMARY  (judge: {report['judge_model']}, reconstruction: {report['reconstruction_model']}, "
          f"snapshot compression: {report['compression']})")
    print(f"  {'mode':9s} {'hallucination':>13s} {'fact recall':>11s} {'mem tokens':>10s} {'returned':>8s} {'dropped':>7s} {'errors':>6s}")
    for mode, s in report["summary"].items():
        recall_text = "-" if s["fact_recall"] is None else f"{s['fact_recall']:.0%}"
        tokens_text = "-" if s["avg_memory_tokens"] is None else f"{s['avg_memory_tokens']:.1f}"
        print(f"  {mode:9s} {s['hallucination_rate']:>13.0%} {recall_text:>11s} {tokens_text:>10s} "
              f"{s['sentences_returned']:>8d} {s['sentences_dropped']:>7d} {s['errors']:>6d}")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"reconstruction_{time.strftime('%Y%m%d_%H%M%S')}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nFull results: {path}")

    if args.demo:
        from app.api.dependencies.qdrant import qdrant_client
        try:
            qdrant_client.delete_collection(settings.SNAPSHOT_COLLECTION)
        except Exception:
            pass


if __name__ == "__main__":
    main()
