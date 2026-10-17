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
- A part the answer marks "Not found in the retrieved text" is an honest gap, not an
  unsupported claim. An answer that says information is missing or was never given is
  supported when the context indeed lacks it.
- Check that the answer is about the same thing the question asks about. If the question
  names a device, case, section, person or document and the answer states facts that the
  context attaches to a DIFFERENT one, set "entity_mismatch" to true and supported to false.
- An answer that only acknowledges or restates what the user said in the Question ("Noted:
  your deadline is 15 November") is supported by the Question itself.
- Sections of the context headed CONVERSATION STATE, HAND-OFF or VERIFIED MEMORY hold what
  was said earlier in this conversation; an answer about the conversation may rely on them.
- List each unsupported statement in "unsupported_claims" (at most 5, each under 25
  words; empty when supported).
- Return ONLY valid JSON.

Example:

{{
    "supported": true,
    "confidence": 0.94,
    "reason": "The answer is fully supported by the retrieved context.",
    "entity_mismatch": false,
    "unsupported_claims": []
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

    if not isinstance(result, dict):

        logger.warning("Verification parsing failed")

        return {

            "supported": False,

            "confidence": 0.0,

            "reason": "Verification parsing failed."

        }

    # A right answer about the wrong thing is not a supported answer.
    if result.get("entity_mismatch") is True:
        result["supported"] = False

    return result
