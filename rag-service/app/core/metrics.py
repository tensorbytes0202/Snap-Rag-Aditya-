"""In-process counters and a per-request LLM-call count.

count_llm_calls() counts the LLM calls made while it is open, including the
ones made on worker threads started with app.utils.parallel.run_parallel
(they share the context). The benchmarks use it for "LLM calls per turn".
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


@contextmanager
def count_llm_calls():
    """Yields {"n": calls so far}; the count is final after the block."""

    box = {"n": 0}
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
