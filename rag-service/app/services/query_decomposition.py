import json
import logging

from app.services.llm_provider import chat


_log = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are an expert query planner.

If the question contains multiple sub-questions,
split it into independent retrieval queries.

Rules:

- Return JSON only
- Maximum 5 subqueries
- If decomposition is unnecessary,
  return the original question only.

Example:

{
    "subqueries": [
        "What is Process Scheduling?",
        "What is Thread Scheduling?",
        "Compare Process Scheduling and Thread Scheduling."
    ]
}
"""


def decompose_query(question: str):

    # ======================================================
    # LLM CALL
    # ======================================================

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

            temperature=0

        )

    except Exception as e:

        _log.warning("Query decomposition failed (%s); falling back to original query", e)

        return [question]

    # ======================================================
    # RESPONSE PARSING
    # ======================================================

    try:

        content = response["message"]["content"].strip()

        # Handle markdown JSON if model returns:
        # ```json
        # {...}
        # ```

        if content.startswith("```"):

            content = content.replace(
                "```json",
                ""
            )

            content = content.replace(
                "```",
                ""
            )

            content = content.strip()

        data = json.loads(content)

        subqueries = data.get(
            "subqueries",
            []
        )

        # Safety checks

        if not isinstance(
            subqueries,
            list
        ):

            return [question]

        subqueries = [
            q.strip()
            for q in subqueries
            if isinstance(q, str)
            and q.strip()
        ]

        if not subqueries:

            return [question]

        # Maximum 5

        return subqueries[:5]

    except Exception as e:

        _log.warning("Query decomposition parsing failed (%s); falling back to original query", e)

        return [question]