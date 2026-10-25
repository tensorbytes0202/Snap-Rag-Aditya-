"""Keeps a benchmark going across days on a free key with a daily request limit.

A free Gemini key allows 500 requests a day per model. Each benchmark is resumable, so this loop runs it, and when the
quota is used up (the run stops or fails) waits and runs it again until everything is done:

    python -m benchmarks.queue_runs docqa       # the architectures on all 75 questions, model from .env
    python -m benchmarks.queue_runs longconv    # the long-conversation variants, on a different model (its own quota)

Run each in its own window; they use different models, so their daily limits do not compete.
"""

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "benchmarks", "results")
PYTHON = sys.executable

DOCQA_ARCHS = ("hybrid_rerank_k5,snaprag,snaprag_fixed_k,abrag,ircot,hybrid_k5,bm25_k5,hybrid_rerank_k10,hyde,crag,adaptive_rag,self_rag,"
               "closed_book,dense_k5,snaprag_retrieval_planner,snaprag_retrieval,snaprag_fixed_k_noplan,snaprag_calib")
LONGCONV_VARIANTS = "full,no_memory,archive,snapshot,evidence,verify,full_fixed_k,full_fixed_interval,full_fixed_lambda,full_no_exemplars,full_history,summary_buffer"
WAIT_MINUTES = 45
MAX_DAYS = 10


def load(name: str):

    path = os.path.join(RESULTS, name)

    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def docqa_done() -> bool:

    data = load("docqa_full75.json")

    if not data:
        return False

    have = {(r["arch"], r["id"]) for r in data["rows"] if "error" not in r}
    ids = data["meta"]["ids"]

    return all((arch, i) in have for arch in DOCQA_ARCHS.split(",") for i in ids)


def longconv_done() -> bool:

    data = load("longconv_part1.json")

    return bool(data) and {r["variant"] for r in data["runs"]} >= set(LONGCONV_VARIANTS.split(","))


def quota_open(model: str) -> bool:
    """One tiny request: False when the model's daily quota is used up (HTTP 429), so no run is started just to fail."""

    import httpx
    from dotenv import dotenv_values

    env = dotenv_values(os.path.join(ROOT, "..", ".env"))
    reply = httpx.post(env["LLM_GATEWAY_URL"].rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + env["LLM_GATEWAY_TOKEN"]},
                       json={"model": model, "messages": [{"role": "user", "content": "ok"}], "reasoning_effort": "low"}, timeout=90)

    return reply.status_code != 429


def quota_open_or_offline(model: str) -> bool:
    """quota_open, but a lost network connection means "try again later" instead of ending the driver."""

    try:
        return quota_open(model)
    except Exception:
        return False


def running(fragment: str) -> bool:
    """Is a benchmark process of this kind already running (started by hand)?"""

    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process | Where-Object {{ $_.Name -eq 'python.exe' -and $_.CommandLine -match '{fragment}' }} | Measure-Object).Count"],
                         capture_output=True, text=True).stdout.strip()

    return out.isdigit() and int(out) > 0


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=("docqa", "longconv"))
    args = parser.parse_args()
    deadline = time.time() + MAX_DAYS * 86400

    while time.time() < deadline:

        if args.which == "docqa":
            if docqa_done():
                break
            command = [PYTHON, "-m", "benchmarks.docqa.run", "--resume", os.path.join(RESULTS, "docqa_full75.json"), "--workers", "4", "--archs", DOCQA_ARCHS]
            env = {**os.environ, "LLM_REQUESTS_PER_MINUTE": "12", "PYTHONUNBUFFERED": "1"}
            fragment = "benchmarks.docqa.run"
        else:
            if longconv_done():
                break
            part = os.path.join(RESULTS, "longconv_part1.json")
            command = [PYTHON, "-m", "benchmarks.longconv.run", "--variants", LONGCONV_VARIANTS, "--turns", "80", "--repeats", "1", "--retries", "1",
                       "--keep-chats", os.path.join(RESULTS, "chats"), "--out", part] + (["--resume", part] if os.path.exists(part) else [])
            env = {**os.environ, "LLM_MODEL": "gemini-3.1-flash-lite", "LLM_REQUESTS_PER_MINUTE": "12", "PYTHONUNBUFFERED": "1"}
            fragment = "benchmarks.longconv.run"

        model = env.get("LLM_MODEL") or "gemini-3.5-flash-lite"

        if running(fragment):
            print(time.strftime("%H:%M"), "a run is already going; waiting", flush=True)
        elif not quota_open_or_offline(model):
            print(time.strftime("%H:%M"), model, "daily quota used up or no network; waiting", flush=True)
        else:
            print(time.strftime("%H:%M"), "starting", " ".join(command[2:6]), flush=True)
            with open(os.path.join(RESULTS, f"queue_{args.which}.log"), "a", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)

        time.sleep(WAIT_MINUTES * 60)

    print(time.strftime("%H:%M"), args.which, "finished" if (docqa_done() if args.which == "docqa" else longconv_done()) else "gave up after the maximum days", flush=True)


if __name__ == "__main__":
    main()
