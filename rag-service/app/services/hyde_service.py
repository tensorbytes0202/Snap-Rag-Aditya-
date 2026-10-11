import logging

from app.core.exceptions import LLMProviderError
from app.services.llm_provider import chat


logger = logging.getLogger(__name__)


def generate_hyde(question: str):

    prompt = f"""
Generate a concise hypothetical passage that would perfectly answer the question.

Question:
{question}

Return only the passage.
"""

    try:

        response = chat(

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return response["message"]["content"].strip()

    except LLMProviderError:

        logger.warning("HyDE generation failed; falling back to raw question")

        return question
