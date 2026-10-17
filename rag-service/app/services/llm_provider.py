import logging

from langsmith.wrappers import wrap_openai
from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.metrics import record_llm_call

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
# max_retries=0: the gateway itself retries transient failures
# (429/5xx/timeouts) with its own backoff, so a second retry layer
# here would only multiply worst-case latency. The timeout must
# comfortably exceed the gateway's own worst case per request.
# ==========================================================

_client = None


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
                    max_retries=0,
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

    try:

        client = _get_client()

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=stream,
            **reasoning_kwargs,
        )

    except LLMProviderError:
        raise

    except Exception as exc:
        raise LLMProviderError(str(exc)) from exc

    if stream:
        return completion

    _log.debug("LLM usage: %s", getattr(completion, "usage", None))

    return {
        "message": {
            "content": completion.choices[0].message.content
        }
    }
