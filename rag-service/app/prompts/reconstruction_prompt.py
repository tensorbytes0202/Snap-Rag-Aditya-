# ==========================================================
# Reconstruction & Judge Prompts
# ==========================================================
#
# Reconstruction expands a lossy snapshot back into sentences, using the
# document pages the snapshot is anchored to for detail. The judge checks
# sentences against labelled sources; the verification cascade and the
# benchmark both use it, with different sources (and models).

RECONSTRUCT_PROMPT = """You rebuild conversation memory from a compressed snapshot.

SNAPSHOT of turns {from_turn}-{to_turn} of a conversation about documents:
{snapshot}
{evidence}
TASK: Write what this memory says that is relevant to: {focus}
- Short, self-contained sentences. Name the subject in each; keep numbers, names and
  units exactly as in the snapshot.
- Use the document passages only to restore detail around snapshot facts.
- If nothing is relevant, return an empty list.

Return ONLY JSON: {{"sentences": ["...", "..."]}}"""


JUDGE_PROMPT = """Check each SENTENCE against the SOURCES below.
For each sentence give the label of the FIRST source (in the order listed) that clearly
states it, or "none" if no source states it or a source contradicts it.
Paraphrases count; a different number, name or claim does not.

{sources}

SENTENCES:
{sentences}

"source" must be exactly one of {labels}: a source label, never a turn or page reference.
Return ONLY JSON: {{"checks": [{{"i": <sentence number>, "source": <label>}}]}}"""


def render_snapshot_for_prompt(snapshot: dict) -> str:

    facets = snapshot["facets"]

    # Plain "(turn n)" markers: bracketed ones get echoed back by the judge
    # in place of the source label.
    lines = [f"- {fact['text']} (turn {fact['turn']})" for fact in facets["facts"]]

    lines += [f"User said (turn {u['turn']}): {u['text']}" for u in facets.get("user_facts", [])]

    if facets["entities"]:
        lines.append("Entities: " + ", ".join(entity["name"] for entity in facets["entities"]))

    lines += [f"Decision ({d['status']}, turn {d['turn']}): {d['topic']}" for d in facets["decisions"]]
    lines += [f"Open question: {q}" for q in facets["open_questions"]]
    lines += [f"The user asked at turn {a['turn']}: {a['text']}" for a in facets.get("asked", [])]

    return "\n".join(lines) or "(empty)"


def render_evidence(evidence: dict) -> str:
    """{(document, page): text} -> labelled passages."""

    return "\n\n".join(f"Page {page} of {document}:\n{text}" for (document, page), text in evidence.items())


def render_archive(turns: list[dict]) -> str:
    """The verbatim turns a snapshot replaced."""

    return "\n\n".join(f"Turn {t['turn']}:\nQ: {t['question']}\nA: {t['answer']}" for t in turns) or "(empty)"


def build_reconstruct_prompt(snapshot: dict, evidence: dict, question: str | None) -> str:

    evidence_block = (
        f"\nDOCUMENT PASSAGES (pages the snapshot cites):\n{render_evidence(evidence)}\n"
        if evidence else ""
    )

    return RECONSTRUCT_PROMPT.format(
        from_turn=snapshot["from_turn"],
        to_turn=snapshot["to_turn"],
        snapshot=render_snapshot_for_prompt(snapshot),
        evidence=evidence_block,
        focus=f'"{question}"' if question else "the whole conversation",
    )


def build_judge_prompt(sentences: list[str], sources: dict[str, str]) -> str:

    labels = ", ".join(f'"{label}"' for label in [*sources, "none"])

    return JUDGE_PROMPT.format(
        sources="\n\n".join(f'=== SOURCE "{label}" ===\n{text}' for label, text in sources.items()),
        sentences="\n".join(f"{i}. {sentence}" for i, sentence in enumerate(sentences, 1)),
        labels=labels,
    )
