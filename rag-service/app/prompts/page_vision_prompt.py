# ==========================================================
# Page Vision Prompt
# ==========================================================
#
# Turns one rendered PDF page into plain text for indexing. OCR reads
# the pixels but loses what they mean: which window a screenshot is,
# which device a value belongs to, how a diagram is wired. A vision
# model keeps that, so retrieval can find e.g. "Router2's static
# routes" even when the page itself has no "Router-2" heading.

PAGE_VISION_PROMPT = """You are converting one page of a PDF into searchable text for a document QA system.
Transcribe and describe EVERYTHING the page shows, especially screenshots, diagrams and tables.

Rules:
- Keep everything in reading order, top to bottom. Name each screenshot by what it shows
  (window title, device, tab), e.g. "Router1 > Config > Static Routes", and trust the window
  title over nearby headings.
- Headings often sit on the page BEFORE their content. If the previous page ended with a
  heading, start with "Continued section: <that heading>". A heading at the very bottom of
  this page, with nothing after it, introduces the NEXT page: keep it last, as
  "Next section: <heading>", and never attach it to this page's screenshots.
- Copy every value exactly: numbers, addresses, names, settings, status messages, table rows.
- For diagrams, list every element and every connection between elements, with the labels on each side.
- For tables or logs, give each row as a line.
- Plain text, no markdown tables, no commentary, no guessing beyond what is visible.
- If the page shows any test, check, status or output (a ping, a log line, a result table),
  add a line "Verification result: <what was tested, the outcome, and what it confirms or
  proves>", e.g. "Verification result: the ping from A to B succeeded, confirming the setup works".
- End with one line starting "Summary:" that says in plain words what the page shows or
  demonstrates.
- Output only the page content. Never repeat the CONTEXT section below or its labels.

CONTEXT (for understanding only, do not copy):
<page_text_layer>{page_text}</page_text_layer>
<previous_page_ending>{previous_text}</previous_page_ending>"""


def build_page_vision_prompt(page_text: str, previous_text: str) -> str:

    return PAGE_VISION_PROMPT.format(
        page_text=page_text or "(none)",
        previous_text=previous_text or "(none)",
    )
