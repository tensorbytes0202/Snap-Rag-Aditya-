import logging

from app.services.llm_provider import chat
from app.utils.json_utils import extract_json


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
You are a Query Classification Engine.

Classify the user's question into ONLY ONE category.

Categories:

DEFINITION
EXPLANATION
COMPARISON
LIST
PROCEDURE
SUMMARY
CODE
FACTUAL

Return ONLY valid JSON.

Example:

{
    "query_type": "DEFINITION"
}
"""


VALID_QUERY_TYPES = {
    "DEFINITION",
    "EXPLANATION",
    "COMPARISON",
    "LIST",
    "PROCEDURE",
    "SUMMARY",
    "CODE",
    "FACTUAL",
}


def classify_query(question: str):

    # ======================================================
    # LLM CLASSIFICATION
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

    except Exception:

        logger.warning("Query classification failed, falling back to EXPLANATION", exc_info=True)

        return "EXPLANATION"

    # ======================================================
    # PARSE RESPONSE
    # ======================================================

    content = response["message"]["content"].strip()

    result = extract_json(content)

    if not isinstance(result, dict):

        logger.warning("Query classification parsing failed, falling back to EXPLANATION")

        return "EXPLANATION"

    query_type = str(
        result.get(
            "query_type",
            "EXPLANATION"
        )
    ).upper().strip()

    if query_type not in VALID_QUERY_TYPES:

        return "EXPLANATION"

    return query_type
