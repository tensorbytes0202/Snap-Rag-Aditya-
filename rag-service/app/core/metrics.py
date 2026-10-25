"""In-process counters and a per-request LLM-call count.

count_llm_calls() counts the LLM calls made while it is open (and their prompt
and completion tokens, and the seconds spent waiting for the request-rate
limit), including the ones made on worker threads started with
app.utils.parallel.run_parallel (they share the context). The benchmarks use
it for "LLM calls per turn", tokens, and latency without the rate-limit wait.
"""

import contextvars
import threading
from contextlib import contextmanager


_lock = threading.Lock()

_counters: dict[tuple, float] = {}

_call_box: contextvars.ContextVar = contextvars.ContextVar("llm_call_box", default=None)


def inc(name: str, value: float = 1, **labels) -> None:

    key = (name, tuple(sorted(labels.items())))

    with _lock:
        _counters[key] = _counters.get(key, 0.0) + value


def record_llm_call(kind: str = "chat") -> None:

    inc("snaprag_llm_calls_total", kind=kind)

    box = _call_box.get()

    if box is not None:
        with _lock:
            box["n"] += 1


def record_usage(prompt_tokens, completion_tokens) -> None:
    """Token counts the provider reported for one call (None when it reported none)."""

    box = _call_box.get()

    if box is not None:
        with _lock:
            box["prompt_tokens"] += prompt_tokens or 0
            box["completion_tokens"] += completion_tokens or 0


def record_wait(seconds: float) -> None:
    """Seconds one call spent waiting for the request-rate limit or a quota retry."""

    box = _call_box.get()

    if box is not None:
        with _lock:
            box["wait"] += seconds


@contextmanager
def count_llm_calls():
    """Yields {"n", "prompt_tokens", "completion_tokens", "wait"} so far; final after the block."""

    box = {"n": 0, "prompt_tokens": 0, "completion_tokens": 0, "wait": 0.0}
    token = _call_box.set(box)

    try:
        yield box
    finally:
        _call_box.reset(token)


def value(name: str, **labels) -> float:

    with _lock:
        return _counters.get((name, tuple(sorted(labels.items()))), 0.0)


def reset() -> None:
    """For tests."""

    with _lock:
        _counters.clear()
