# ==========================================================
# Snapshot Prompt
# ==========================================================
#
# The snapshot engine's encoder: turns a window of chat turns into the
# four facets of the SnapRAG design. Numbers, names and units are kept
# verbatim because compression loses them first, and every fact names
# its subject so it still makes sense once the turns are gone.

import json


SNAPSHOT_PROMPT = """You compress part of a conversation about documents into a memory snapshot.
The snapshot replaces these turns later, so anything you leave out is forgotten.

Return ONLY JSON in this shape:
{{
  "facts": [{{"text": "...", "turn": <turn number>}}],
  "entities": [{{"name": "...", "turns": [<turn numbers>]}}],
  "decisions": [{{"topic": "...", "status": "closed" or "open", "turn": <turn number>}}],
  "open_questions": ["..."]
}}

Rules:
- facts: atomic statements taken from the ANSWERS. Keep every number, address, name,
  date and unit exactly as written. Name the subject in each fact ("Router-2 routes
  10.0.0.0/8 via 50.50.50.1"), never "it" or "this". One statement per fact, no duplicates.
- Skip answers that say information was missing or could not be verified.
- entities: the named things the facts are about (devices, people, concepts, documents).
- decisions: conclusions or choices the user reached; "open" if still unsettled.
- open_questions: questions the user asked that the answers did not resolve.
- Use only what the turns say. Do not add outside knowledge.
{exemplars}
TURNS:
{turns}"""


def build_snapshot_prompt(window: list[dict], exemplar_facets: list[dict]) -> str:

    exemplars = ""

    if exemplar_facets:
        exemplars = (
            "\nEARLIER SNAPSHOTS of this conversation (follow their style and entity names):\n"
            + "\n".join(json.dumps(facets, ensure_ascii=False) for facets in exemplar_facets)
            + "\n"
        )

    turns = "\n\n".join(
        f"[turn {turn['turn']}]\nQ: {turn['question']}\nA: {turn['answer']}"
        for turn in window
    )

    return SNAPSHOT_PROMPT.format(exemplars=exemplars, turns=turns)
