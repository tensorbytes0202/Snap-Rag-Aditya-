import logging
import re
import threading
import time
from collections import deque

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.metrics import record_llm_call, record_usage, record_wait

_log = logging.getLogger(__name__)


# ==========================================================
# Config
# ==========================================================

DEFAULT_MODEL = settings.LLM_MODEL


# ==========================================================
# Lazy Client
# ==========================================================
#
# The client must NOT be constructed at import time: doing so
# eagerly requires the gateway token just to import this module
# (and therefore main.py), which breaks startup for Docker before
# secrets are wired up. It's built on first use and cached; a
# construction failure is reported as an LLMProviderError, same as
# any other provider failure.
#
# Transient failures (429, 5xx, timeouts) are retried by the SDK with backoff,
# up to LLM_MAX_RETRIES. The timeout must comfortably exceed one slow request.
# ==========================================================

_client = None

# Requests started in the last minute, for LLM_REQUESTS_PER_MINUTE (the free
# Gemini tier allows 15 a minute per model; going over fails the request).
_started: deque = deque()
_started_lock = threading.Lock()
_RETRY_IN = re.compile(r"retry in ([0-9.]+)s")


def _wait_for_slot() -> None:
    """Blocks until a request may start without passing the per-minute limit."""

    limit = settings.LLM_REQUESTS_PER_MINUTE
    started = time.monotonic()

    while limit > 0:

        with _started_lock:

            now = time.monotonic()

            while _started and now - _started[0] >= 60:
                _started.popleft()

            if len(_started) < limit:
                _started.append(now)
                record_wait(now - started)
                return

            wait = 60 - (now - _started[0])

        time.sleep(wait + 0.1)


def _retry_delay(exc: Exception, attempt: int) -> float | None:
    """Seconds to wait before trying again, or None when trying again cannot help: a 429 waits as long as
    the provider asks (a per-day quota is not retried), a server error, timeout or lost connection backs off
    exponentially, any other error is final."""

    text, status = str(exc), getattr(exc, "status_code", None)

    if status == 429 or text.startswith("Error code: 429"):
        if "PerDay" in text:
            return None
        match = _RETRY_IN.search(text)
        return min(float(match.group(1)) + 1, 90) if match else min(60.0, 5.0 * 2 ** attempt)

    if status is None or status >= 500:
        return min(30.0, 2.0 * 2 ** attempt)

    return None


def _get_client():

    global _client

    if _client is None:

        try:
            # wrap_openai makes every chat.completions call show up as
            # a child run (prompt, response, tokens, latency) of
            # whichever @traceable function is running. It is a no-op
            # when LANGSMITH_TRACING is not enabled.
            _client = wrap_openai(
                OpenAI(
                    api_key=settings.LLM_GATEWAY_TOKEN,
                    base_url=settings.LLM_GATEWAY_URL,
                    timeout=settings.LLM_REQUEST_TIMEOUT_SECONDS,
                    max_retries=0,           # retries are made in chat(), so each one passes the request-rate limit
                )
            )
        except Exception as exc:
            raise LLMProviderError(str(exc)) from exc

    return _client


# ==========================================================
# Universal Chat Function
# ==========================================================

def chat(
    messages,
    stream=False,
    temperature=0,
    model=None,
    **kwargs
):

    model_name = model or DEFAULT_MODEL

    # Reasoning effort is gemini-specific. An empty string means
    # "don't send the parameter" -- some models (e.g. a "minimal"
    # level on gemini-3.8-flash) reject it outright. The gateway
    # forwards unknown fields as-is to whichever provider a request
    # resolves to.
    reasoning_kwargs = {}

    if settings.LLM_REASONING_EFFORT:
        reasoning_kwargs["reasoning_effort"] = settings.LLM_REASONING_EFFORT

    record_llm_call("stream" if stream else "chat")

    for attempt in range(settings.LLM_MAX_RETRIES + 1):

        _wait_for_slot()

        try:

            client = _get_client()

            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                stream=stream,
                **reasoning_kwargs,
            )

            break

        except LLMProviderError:
            raise

        except Exception as exc:

            delay = _retry_delay(exc, attempt)

            if delay is None or attempt == settings.LLM_MAX_RETRIES:
                raise LLMProviderError(str(exc)) from exc

            record_wait(delay)
            time.sleep(delay)

    if stream:
        return completion

    usage = getattr(completion, "usage", None)
    record_usage(getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))

    return {
        "message": {
            "content": completion.choices[0].message.content
        }
    }
