import json
import logging

from app.services.llm_provider import chat


_log = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Generate ONLY 2 semantic rewrites of the user's query.

Rules:

- Preserve meaning
- Different wording
- No explanation
- Return ONLY JSON

Example:

{
    "queries": [
        "...",
        "..."
    ]
}
"""


def rewrite_query(question: str) -> list[str]:

    try:

        response = chat(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.2
        )

        content = response["message"]["content"]

        data = json.loads(content)

        queries = data.get("queries", [])

        if not queries:
            return [question]

        return queries[:2]

    except Exception as exc:

        _log.warning("Query rewrite failed (%s); falling back to original query", exc)

        return [question]