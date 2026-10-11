import json
import logging

from app.services.llm_provider import chat


_log = logging.getLogger(__name__)


def _fallback_self_query(question: str):
    """
    Safe fallback when LLM-based self-query generation fails.
    """
    return {
        "query": question,
        "document_name": None,
        "page": None,
        "chapter": None,
        "section": None,
        "topic": None,
    }


def generate_self_query(question: str):

    prompt = f"""
You are an intelligent query parser.

Extract structured search filters from the user's question.

Return ONLY valid JSON.

Schema:

{{
    "query": "",
    "document_name": null,
    "page": null,
    "chapter": null,
    "section": null,
    "topic": null
}}

Rules:

- Preserve the original meaning of the question.
- "query" should contain the actual semantic search query.
- Only extract a filter when it is explicitly present.
- Do not guess document names, pages, chapters, or sections.
- If a field is not present, return null.

Question:

{question}
"""

    # ======================================================
    # LLM CALL
    # ======================================================

    try:

        response = chat(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

    except Exception as e:

        _log.warning("Self-query LLM call failed (%s); using fallback self-query", e)

        return _fallback_self_query(question)

    # ======================================================
    # RESPONSE PARSING
    # ======================================================

    try:

        content = response["message"]["content"].strip()

        # Handle accidental markdown JSON:
        if content.startswith("```"):

            content = content.replace("```json", "")
            content = content.replace("```", "")
            content = content.strip()

        data = json.loads(content)

        # Make sure required fields exist
        result = {
            "query": data.get("query") or question,
            "document_name": data.get("document_name"),
            "page": data.get("page"),
            "chapter": data.get("chapter"),
            "section": data.get("section"),
            "topic": data.get("topic"),
        }

        return result

    except Exception as e:

        _log.warning("Self-query parsing failed (%s); using fallback self-query", e)

        return _fallback_self_query(question)