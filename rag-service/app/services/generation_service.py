from langsmith import traceable
import logging

from app.services.llm_provider import chat
from app.prompts.generation_prompt import build_generation_prompt


_log = logging.getLogger(__name__)


# ============================================================
# Normal Generation
# ============================================================

@traceable(name="generate_answer", run_type="chain")
def generate_answer(
    question: str,
    context: str,
    query_type: str
):

    prompt = build_generation_prompt(
        question,
        context,
        query_type
    )

    response = chat(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    _log.debug("Generated answer: %s", answer)

    return answer


# ============================================================
# Streaming Generation
# ============================================================

@traceable(name="stream_answer", run_type="chain")
def stream_answer(
    question: str,
    context: str,
    query_type: str
):

    prompt = build_generation_prompt(
        question,
        context,
        query_type
    )

    stream = chat(

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        stream=True

    )

    # The gateway's chat/completions endpoint is OpenAI-compatible,
    # so every streamed chunk has this shape regardless of which
    # provider the model resolves to.
    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta and delta.content:

            yield delta.content