from langsmith import traceable
import logging

from app.services.llm_provider import chat


_log = logging.getLogger(__name__)


@traceable(name="generate_followups", run_type="chain")
def generate_followups(question, answer):

    try:

        prompt = f"""
You are an expert teacher.

User Question:
{question}

Answer:
{answer}

Generate exactly 3 follow-up questions.

Rules:

- Questions only
- One per line
- No numbering
- No explanation
"""

        response = chat(

            messages=[

                {
                    "role": "user",
                    "content": prompt
                }

            ]

        )

        text = response["message"]["content"]

        questions = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            line = line.lstrip("1234567890.- ")

            questions.append(line)

        return questions[:3]

    except Exception as e:

        _log.warning("Follow-up generation failed: %s", e)

        return []