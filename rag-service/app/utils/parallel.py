"""Run independent blocking calls side by side.

Each task runs in a copy of the caller's context, so LangSmith's parent run
and the LLM-call counter (app.core.metrics.count_llm_calls) follow it into
the worker thread.
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor


def run_parallel(*tasks):
    """Results of the zero-argument callables, in order. All tasks finish
    first; then the first exception, if any, is raised."""

    if len(tasks) == 1:
        return [tasks[0]()]

    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, task) for task in tasks]

    return [future.result() for future in futures]
