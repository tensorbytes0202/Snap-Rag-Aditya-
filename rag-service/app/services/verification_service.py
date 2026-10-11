from langsmith import traceable
import logging

from app.services.llm_provider import chat
from app.utils.json_utils import extract_json


logger = logging.getLogger(__name__)


# ==========================================================
# Verify Generated Answer
# ==========================================================

@traceable(name="verify_answer", run_type="chain")
def verify_answer(
    question: str,
    answer: str,
    context: str
):

    prompt = f"""
You are an expert RAG verification system.

Your job is to verify whether the generated answer is supported by the retrieved context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Instructions:

- Compare ONLY with the retrieved context.
- Do NOT use outside knowledge.
- Conclusions that follow directly from the context count as supported, e.g. a path
  traced through connections the context lists, or a result the context shows.
- If the answer is fully supported, return supported=true.
- If any stated fact is missing from or contradicts the context, supported=false.
- Return ONLY valid JSON.

Example:

{{
    "supported": true,
    "confidence": 0.94,
    "reason": "The answer is fully supported by the retrieved context."
}}
"""

    response = chat(

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0

    )

    content = response["message"]["content"].strip()

    result = extract_json(content)

    if result is None:

        logger.warning("Verification parsing failed")

        return {

            "supported": False,

            "confidence": 0.0,

            "reason": "Verification parsing failed."

        }

    return result
