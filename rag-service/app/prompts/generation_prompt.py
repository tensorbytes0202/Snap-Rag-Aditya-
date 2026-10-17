# ==========================================================
# Formatting Instructions
# ==========================================================

def get_format_instruction(query_type):

    instructions = {

        "COMPARISON": """
Answer in a comparison table.

Columns: Feature, then one column for each item being compared, headed with the item's
actual name (never "Item 1" or "Item 2"). Put each item's own facts in its own column. When
the text has nothing on a feature for an item, write "Not found in the retrieved text" in that
cell; never leave placeholder text such as <...> in a cell.
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
""",

        "CONVERSATION": """
Answer briefly and directly, in one or two sentences or a short list.
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
CONVERSATION SO FAR (to understand what the question refers to, e.g. "it", "that router",
"what about X", and to recall what was asked or said; facts about the documents must still
come from CONTEXT; long answers are shortened here):
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
4. If the question has several parts, answer every part the context supports and
   write "Not found in the retrieved text: <what is missing>" for each part it does
   not. Reply exactly "Insufficient information found." only when the context
   supports no part of the question.
5. Every passage starts with [Source: document, PDF page N]. When you state a rule
   or name a case, cite its page as (PDF p. N) using only that N, never a number
   printed inside the passage text. If the passage's running header visibly shows
   the book's own page number, you may add it as (printed p. M). Never cite a page,
   section number, case name or quotation that does not appear in the context.
6. Keep what the context states apart from your own application of it to the
   question's facts (e.g. "The text states:" / "Applying this:").
7. Follow the formatting instructions strictly.
8. Do not hallucinate. Do not invent facts.
9. A "CONVERSATION STATE", "HAND-OFF" or "VERIFIED MEMORY" section, if present, holds what
   was said earlier in this conversation. Use it only to recall what was said before; do not
   present it as the document's content, and do not mention it otherwise.
10. A question about what the user said, asked or decided earlier in this conversation
   ("what is my deadline?", "which sections did I skip?", "which case did I ask about at turn 5?")
   is answered from those sections and from CONVERSATION SO FAR, never from the documents. If
   they do not contain it, reply "You have not told me that in this conversation." Do not
   guess it, and do not answer such a question with document content.

FORMAT INSTRUCTION:
{format_instruction}
{_format_history(history)}
CONTEXT:
{context}

QUESTION:
{question}
"""