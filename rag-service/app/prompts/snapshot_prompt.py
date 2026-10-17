# ==========================================================
# Snapshot Prompt
# ==========================================================
#
# The snapshot engine's encoder: turns a window of chat turns into the
# facets of the SnapRAG design. Numbers, names and units are kept
# verbatim because compression loses them first, and every fact names
# its subject so it still makes sense once the turns are gone.
#
# What the USER said (their deadlines, choices, constraints) is kept apart
# from what the ANSWERS said about the documents: user statements are not in
# any document, so once the turn is gone this is the only place they survive.

import json


SNAPSHOT_PROMPT = """You compress part of a conversation about documents into a memory snapshot.
The snapshot replaces these turns later, so anything you leave out is forgotten.

Return ONLY JSON in this shape:
{{
  "facts": [{{"text": "...", "turn": <turn number>}}],
  "user_facts": [{{"text": "...", "turn": <turn number>}}],
  "entities": [{{"name": "...", "turns": [<turn numbers>]}}],
  "decisions": [{{"topic": "...", "status": "closed" or "open", "turn": <turn number>}}],
  "open_questions": ["..."]
}}

Rules:
- facts: atomic statements taken from the ANSWERS about the documents. Keep every number,
  address, name, date, case name, section number, page and unit exactly as written. Name the
  subject in each fact ("Router-2 routes 10.0.0.0/8 via 50.50.50.1"), never "it" or "this".
  One statement per fact, no duplicates.
- Skip answers that say information was missing or could not be verified.
- user_facts: things the USER stated in their own messages that are not from a document:
  deadlines, dates, names, limits, preferences, constraints, the example or scope they chose,
  corrections to earlier statements ("actually I will also cover X"). Copy the user's wording and
  numbers exactly and write each as a full statement ("The user's draft deadline is 15 November").
- decisions: only choices or commitments the USER made about their own task ("report will cover
  only sections 26 and 27, not 32 and 33"); "open" if they are still undecided. Never list a topic
  just because it was explained or asked about.
- entities: the named things the facts are about (devices, people, cases, sections, concepts).
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
