# ==========================================================
# Formatting Instructions
# ==========================================================

def get_format_instruction(query_type):

    instructions = {

        "COMPARISON": """
Answer in a comparison table.

Columns:
Feature | Item 1 | Item 2
""",

        "ADVANTAGES": """
Answer using bullet points.
""",

        "DISADVANTAGES": """
Answer using bullet points.
""",

        "STEPS": """
Answer step-by-step using numbered points.
""",

        "LIST": """
Answer using bullet points.
""",

        "EXPLANATION": """
Give a detailed explanation using headings and bullet points.
"""

    }

    return instructions.get(
        query_type,
        "Give a concise answer."
    )


# ==========================================================
# Prompt Builder
# ==========================================================

def _format_history(history) -> str:
    """Recent chat turns, oldest first. Empty without history, so a
    question asked outside a chat gets exactly the same prompt as before."""

    if not history:
        return ""

    lines = "\n".join(
        f"Q{turn['turn']}: {turn['question']}\nA{turn['turn']}: {turn['answer']}"
        for turn in history
    )

    return f"""
CONVERSATION SO FAR (only to understand what the question refers to, e.g. "it",
"that router", "what about X"; facts must still come from CONTEXT):
{lines}
"""


def build_generation_prompt(

    question,
    context,
    query_type,
    history=None,

):

    format_instruction = get_format_instruction(
        query_type
    )

    return f"""
You are an Advanced RAG Assistant.

Rules:

1. Answer ONLY from the provided context.
2. Do NOT use outside knowledge.
3. An answer that follows directly from the context counts as found: combine facts
   from several passages, trace a path through connections the context lists, or say
   what a result shown in the context proves.
   Only if the context lacks the facts needed, reply exactly:
   Insufficient information found.
4. Follow the formatting instructions strictly.
5. Do not hallucinate.
6. Do not invent facts.
7. Use only the retrieved context.

FORMAT INSTRUCTION:
{format_instruction}
{_format_history(history)}
CONTEXT:
{context}

QUESTION:
{question}
"""