from app.services.chat_memory import get_history


# ==========================================================
# Conversation History
# ==========================================================

def build_history():

    history = get_history()

    history_text = ""

    for msg in history[-6:]:

        history_text += (
            f"{msg['role']}: {msg['content']}\n"
        )

    return history_text


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

def build_generation_prompt(

    question,
    context,
    query_type

):

    history_text = build_history()

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

CONVERSATION HISTORY:
{history_text}

CONTEXT:
{context}

QUESTION:
{question}
"""