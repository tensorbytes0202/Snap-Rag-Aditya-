"""Long-conversation document-QA benchmark (SnapRAG-LongConv).

Plays a scripted conversation (100 turns by default) about the torts book
against each system in turn: document questions fill the context, the user
plants facts about their own coursework, and later probes ask for them. See
script.py for the turn mix and variants.py for the systems compared.

Run from rag-service/ (needs Qdrant with the book indexed, and the LLM gateway):

    ..\\venv\\Scripts\\python.exe -m benchmarks.longconv.run --list
    ..\\venv\\Scripts\\python.exe -m benchmarks.longconv.run --variants main --turns 100 --repeats 3
    ..\\venv\\Scripts\\python.exe -m benchmarks.longconv.run --variants full,no_memory --turns 80   # shortest script
    ..\\venv\\Scripts\\python.exe -m benchmarks.longconv.run --dry-run                             # no services at all

A run costs about turns x 4 LLM calls per variant (more with memory upkeep),
so start with a short script. Results are saved after every variant; rerun
with --resume <file> to continue an interrupted run.
"""

import argparse
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from benchmarks.atomic import dump_json  # noqa: E402
from benchmarks.longconv import report, scoring  # noqa: E402
from benchmarks.longconv import variants as variant_defs  # noqa: E402
from benchmarks.longconv.dryrun import DryRunSystem  # noqa: E402
from benchmarks.longconv.script import build_script, load_spec  # noqa: E402

USER = "benchmark-user"

SUMMARY_PROMPT = """You keep a running summary of a conversation between a user and an assistant about a document.

CURRENT SUMMARY:
{summary}

NEXT TURN:
Q: {question}
A: {answer}

Rewrite the summary to include this turn, in at most 250 words. Keep everything the USER said about
themselves or their task (deadlines, names, numbers, choices, corrections) exactly as stated, and the
topics discussed. Return only the summary."""


# ==========================================================
# Conversation: history, recording and upkeep of one variant
# ==========================================================

class SummaryBuffer:
    """Rolling summary of turns older than the last `keep`."""

    def __init__(self, summarize, keep: int = 3):
        self.summarize, self.keep = summarize, keep
        self.text, self.upto = "", 0

    def history(self, turns: list[dict]) -> list[dict]:

        recent = [{"turn": t["turn"], "question": t["question"], "answer": t["answer"][:800]}
                  for t in turns[-self.keep:]]

        if not self.text:
            return recent

        return [{"turn": 0, "question": "Summary of the earlier conversation", "answer": self.text}] + recent

    def update(self, turns: list[dict]) -> None:
        """Fold turns that fell out of the window into the summary."""

        while len(turns) - self.upto > self.keep:
            self.text = self.summarize(self.text, turns[self.upto])
            self.upto += 1


class Conversation:

    def __init__(self, variant, user: str, chat_id: str, summarize=None, upkeep: bool = True):
        self.variant, self.user, self.chat_id = variant, user, chat_id
        self.upkeep_enabled = upkeep
        self.summary = SummaryBuffer(summarize) if variant.history == "summary" else None

    def turns(self) -> list[dict]:
        from app.services.session_store import get_turns
        return get_turns(self.user, self.chat_id)

    def history(self) -> list[dict]:

        if self.summary:
            return self.summary.history(self.turns())

        from app.services.history import prompt_history
        return prompt_history(self.user, self.chat_id)

    def session(self):
        """What process_query takes for memory; None for the baselines."""
        return (self.user, self.chat_id) if self.variant.uses_snapshots else None

    def record(self, question: str, response: dict, document: str) -> None:

        from app.services.session_store import append_turn

        verification = response.get("verification")

        append_turn(
            self.user, self.chat_id, question, response["answer"], document=document,
            citations=response.get("citations", []),
            verified=verification.get("supported") if isinstance(verification, dict) else None,
            confidence=response.get("confidence"),
        )

    def upkeep(self) -> None:
        """What runs in the background after a turn in production."""

        if not self.upkeep_enabled:
            return

        if self.variant.uses_snapshots:
            from app.services.snapshot_engine import after_turn
            after_turn(self.user, self.chat_id)
        elif self.summary:
            self.summary.update(self.turns())

    def snapshot_stats(self) -> dict:

        if not self.variant.uses_snapshots:
            return {"compression": [], "snapshots": 0, "state_items": 0}

        from app.services.memory_state import current_state
        from app.services.snapshot_store import list_snapshots

        snapshots = list_snapshots(self.user, self.chat_id)

        return {
            "compression": [s["meta"].get("compression") for s in snapshots],
            "snapshots": len(snapshots),
            "state_items": len(current_state(self.user, self.chat_id)),
        }


# ==========================================================
# Measuring one turn
# ==========================================================

@contextlib.contextmanager
def capture_prompts():
    """Sizes (estimated tokens) of the answer prompts built while open."""

    from app.prompts.generation_prompt import build_generation_prompt
    from app.services import answering

    sizes = []
    original = answering.generate_answer

    def wrapped(question, context, query_type, history=None):
        sizes.append(len(build_generation_prompt(question, context, query_type, history)) // 4)
        return original(question, context, query_type, history)

    answering.generate_answer = wrapped

    try:
        yield sizes
    finally:
        answering.generate_answer = original


def ask_with_retry(ask, retries: int, wait: float):
    """(response, None) or (None, error). Rate limits come back as LLM errors,
    so they are waited out rather than counted against the system."""

    from app.core.exceptions import LLMProviderError

    error = "unknown"

    for attempt in range(max(1, retries)):

        try:
            return ask(), None
        except LLMProviderError as exc:
            error = str(exc)[:200]

        if attempt < retries - 1:
            time.sleep(wait)

    return None, error


def play(variant, script: list[dict], spec: dict, *, ask, conversation: Conversation, document: str,
         repeat: int = 0, retries: int = 3, wait: float = 30.0, pause: float = 0.0, log=print) -> dict:
    """One conversation of one variant. `ask(question, history, session)`
    returns the response dict. Returns the run record (see report.py)."""

    from app.core.metrics import count_llm_calls

    refusal = spec["refusal_pattern"]
    turns_out, probes_out, failed = [], [], set()
    streak = 0

    for item in script:

        turn, question = item["turn"], item["text"]
        history = conversation.history()

        with count_llm_calls() as answer_calls, capture_prompts() as prompts:
            started = time.perf_counter()
            response, error = ask_with_retry(
                lambda: ask(question, history, conversation.session()), retries, wait
            )
            seconds = time.perf_counter() - started

        row = {"turn": turn, "kind": item["kind"], "seconds": round(seconds, 2), "query_calls": answer_calls["n"],
               "prompt_tokens": sum(prompts) if prompts else None, "maintenance_calls": 0, "error": error}

        streak = streak + 1 if error else 0

        if streak >= 5:
            # A conversation with holes proves nothing; stop instead of burning hours of quota on more failures.
            raise RuntimeError(f"{variant.name}: 5 turns in a row failed ({error[:120]}); stopping so the variant can be rerun")

        if response is None:
            failed.add(turn)
            response = {"answer": "[no answer: the service was unavailable]", "citations": []}
        else:
            memory = response.get("retrieved_memory") or {}
            row.update({
                "confidence": response.get("confidence"),
                "lambda": memory.get("lambda"),
                "memory_sentences": len(memory.get("sentences") or []),
                "state_items": memory.get("state_items"),
                "page_hit": scoring.page_hit(item.get("gold_pages"), response.get("citations")) if item["kind"] == "doc" else None,
                "attempts": len(response.get("attempts") or []),
            })

        # Recorded even after a failure, so chat turn numbers stay equal to script turns.
        conversation.record(question, response, document)

        with count_llm_calls() as upkeep_calls:
            conversation.upkeep()

        row["maintenance_calls"] = upkeep_calls["n"]
        # Provider-reported tokens of the whole turn (answer and memory upkeep), and the seconds
        # of the answer without the wait for the request-rate limit.
        row["tokens_in"] = answer_calls["prompt_tokens"] + upkeep_calls["prompt_tokens"]
        row["tokens_out"] = answer_calls["completion_tokens"] + upkeep_calls["completion_tokens"]
        row["active_seconds"] = round(max(0.0, seconds - answer_calls["wait"]), 2)
        turns_out.append(row)

        if item["kind"] == "probe":
            graded = scoring.score_probe(item, response["answer"], refusal)
            probes_out.append({
                "id": item["probe_id"], "turn": turn, "kind": item["probe_kind"], "bucket": item.get("bucket"),
                "distance": item.get("distance"), "correct": graded["correct"] and error is None,
                "reason": "service unavailable" if error else graded["reason"],
                "excluded": error is not None or any(s in failed for s in item["source_turns"]),
                "answer": response["answer"][:600],
            })

        log(f"  [{variant.name} #{repeat + 1}] turn {turn:3d}/{len(script)} {item['kind']:<10s} {seconds:5.1f}s "
            f"calls {answer_calls['n']}+{upkeep_calls['n']}"
            + (f"  {'OK ' if probes_out[-1]['correct'] else 'MISS'} {probes_out[-1]['reason']}" if item["kind"] == "probe" else "")
            + (f"  ERROR {error}" if error else ""))

        if pause:
            time.sleep(pause)

    stats = conversation.snapshot_stats()

    return {"variant": variant.name, "repeat": repeat, "probes": probes_out, "turns": turns_out,
            "compression": stats["compression"], "snapshots": stats["snapshots"], "state_items": stats["state_items"]}


# ==========================================================
# Whole benchmark
# ==========================================================

@contextlib.contextmanager
def overridden(overrides: dict):
    """Applies `overrides` to the global settings, then restores them."""

    from app.core.config import settings

    saved = {name: getattr(settings, name) for name in overrides}

    for name, value in overrides.items():
        setattr(settings, name, value)

    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(settings, name, value)


@contextlib.contextmanager
def isolated_storage():
    """Chats, memory and the snapshot index go to throw-away places; the
    documents and their indexes are the real ones."""

    with tempfile.TemporaryDirectory(prefix="snaprag_longconv_") as folder:

        collection = f"snapshot_memory_bench_{uuid.uuid4().hex[:8]}"

        with overridden({"MEMORY_ROOT": folder, "SNAPSHOT_COLLECTION": collection}):
            try:
                yield
            finally:
                try:
                    from app.api.dependencies.qdrant import qdrant_client
                    qdrant_client.delete_collection(collection)
                except Exception:
                    pass


@contextlib.contextmanager
def answers_not_cached():
    """The answer cache would replay one variant's answers to the next."""

    from app.services import query_service

    saved = query_service.get_cache, query_service.set_cache
    query_service.get_cache, query_service.set_cache = (lambda key: None), (lambda key, value: None)

    try:
        yield
    finally:
        query_service.get_cache, query_service.set_cache = saved


def live_summarizer(summary: str, turn: dict) -> str:

    from app.services.llm_provider import chat

    prompt = SUMMARY_PROMPT.format(summary=summary or "(empty)", question=turn["question"], answer=turn["answer"][:1500])

    return chat([{"role": "user", "content": prompt}])["message"]["content"].strip()


def run_benchmark(spec: dict, names: list[str], *, turns: int, seed: int, repeats: int, document: str,
                  dry_run: bool, out_path: str | None, resume: dict | None = None, retries: int = 3,
                  wait: float = 30.0, pause: float = 0.0, remembers: int = 40, keep_chats: str | None = None, log=print) -> list[dict]:
    """Every (variant, repeat) once; returns the run records."""

    from app.services.session_store import create_chat

    runs = list(resume["runs"]) if resume else []
    done = {(r["variant"], r["repeat"]) for r in runs}

    for repeat in range(repeats):

        script = build_script(spec, turns, seed + repeat)

        for name in names:

            if (name, repeat) in done:
                log(f"skipping {name} #{repeat + 1} (already in the results)")
                continue

            variant = variant_defs.VARIANTS[name]
            log(f"\n== {name} #{repeat + 1}/{repeats}: {variant.description}")

            if dry_run:
                oracle = DryRunSystem(script, remembers)
                clock = {"turn": 0}

                def ask(question, history, session, oracle=oracle, clock=clock):
                    clock["turn"] += 1
                    return oracle.answer(question, clock["turn"])
            else:
                from app.services.query_service import process_query

                def ask(question, history, session):
                    return process_query(question, document, history, session)

            with overridden({**variant.settings, "FOLLOWUPS_ENABLED": False}), isolated_storage(), answers_not_cached():
                conversation = Conversation(variant, USER, create_chat(USER), None if dry_run else live_summarizer,
                                           upkeep=not dry_run)
                runs.append(play(variant, script, spec, ask=ask, conversation=conversation, document=document,
                                 repeat=repeat, retries=retries, wait=wait, pause=pause, log=log))

                if keep_chats and not dry_run:
                    # The chat and its snapshots, for benchmarks/longconv/ladder.py (the temporary folder is deleted).
                    from app.core.config import settings
                    shutil.copytree(settings.MEMORY_ROOT, os.path.join(keep_chats, f"{name}_r{repeat}"), dirs_exist_ok=True)

            if out_path:
                from app.core.config import settings
                save(out_path, runs, {"turns": turns, "seed": seed, "repeats": repeats, "variants": names,
                                      "dry_run": dry_run, "document": document, "model": settings.LLM_MODEL,
                                      "date": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit()})

    return runs


def git_commit() -> str:
    """The short git commit of the run, with a * when the tree has uncommitted changes."""

    import subprocess

    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
        return head + ("*" if dirty else "")
    except Exception:
        return "unknown"


def save(path: str, runs: list[dict], meta: dict) -> None:

    dump_json({"meta": meta, "runs": runs}, path)


def comparisons(runs: list[dict], reference: str = "no_memory") -> list[str]:
    """Paired accuracy differences of every variant against `reference`."""

    by_variant = {}

    for run in runs:
        by_variant.setdefault(run["variant"], []).append(run)

    if reference not in by_variant:
        return []

    lines = [f"### Paired accuracy difference vs {reference} (same probes, 95% interval)", ""]

    for name, group in by_variant.items():

        if name == reference:
            continue

        diff = report.paired_difference(group, by_variant[reference])

        if diff:
            low, high = diff["ci"]
            lines.append(f"- {name}: {diff['difference']:+.1%}  ({low:+.1%} to {high:+.1%}, {diff['n']} probes)")

    return lines + [""]


def main(argv=None):

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="main", help="variant or group names, comma separated (see --list)")
    parser.add_argument("--turns", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0, help="script seed; repeat r uses seed + r")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--document", help="indexed document name (default: the spec's)")
    parser.add_argument("--out", help="results file (default: benchmarks/results/longconv_<time>.json)")
    parser.add_argument("--resume", help="continue the results file of an interrupted run")
    parser.add_argument("--retries", type=int, default=3, help="attempts per turn when the LLM is rate limited")
    parser.add_argument("--wait", type=float, default=30.0, help="seconds between those attempts")
    parser.add_argument("--pause", type=float, default=0.0, help="seconds to wait between turns")
    parser.add_argument("--keep-chats", help="copy each finished conversation (chat and snapshots) into this folder, for ladder.py")
    parser.add_argument("--dry-run", action="store_true", help="a stand-in system, to test the benchmark itself")
    parser.add_argument("--remembers", type=int, default=40, help="dry run only: turns the stand-in remembers")
    parser.add_argument("--list", action="store_true", help="list the variants and exit")
    args = parser.parse_args(argv)

    spec = load_spec()

    if args.list:
        for name, variant in variant_defs.VARIANTS.items():
            print(f"{name:22s} {variant.description}")
        print("\ngroups: " + "; ".join(f"{g} = {', '.join(v)}" for g, v in variant_defs.GROUPS.items() if g != "all"))
        return 0

    try:
        names = variant_defs.resolve(args.variants)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        os.environ.setdefault("JWT_SECRET_KEY", "dry-run")
        os.environ.setdefault("LLM_GATEWAY_TOKEN", "dry-run")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        load_dotenv(os.path.join(ROOT, "..", ".env"))

    import logging
    logging.disable(logging.CRITICAL)

    resume = json.load(open(args.resume, encoding="utf-8")) if args.resume else None
    out = args.out or args.resume or os.path.join(ROOT, "benchmarks", "results", f"longconv_{time.strftime('%Y%m%d_%H%M%S')}.json")

    runs = run_benchmark(
        spec, names, turns=args.turns, seed=args.seed, repeats=args.repeats, document=args.document or spec["document"],
        dry_run=args.dry_run, out_path=out, resume=resume, retries=args.retries, wait=args.wait, pause=args.pause,
        remembers=args.remembers, keep_chats=args.keep_chats,
    )

    text = report.render_markdown(report.summarize(runs)) + "\n" + "\n".join(comparisons(runs))
    print("\n" + text)

    with open(os.path.splitext(out)[0] + ".md", "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Results: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
