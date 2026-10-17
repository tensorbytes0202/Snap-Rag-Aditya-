"""SMS-AR: recall verified snapshot memory for the current question.

    lambda(t) = sigmoid(beta * (t - theta))   memory weight; rises with turn t

    1. find the chat's snapshots most similar to the question (snapshot
       index): at most SMS_TOP_K, each at least SMS_MIN_SIMILARITY
    2. reconstruct + verify them together in one recall() call; facts
       about a document other than the active one are left out
    3. pack the verified sentences into lambda(t) * MEMORY_CONTEXT_TOKENS,
       skipping duplicates and sentences the document context already has

The conversation state (what the user stated or decided, see memory_state)
does not compete for those slots: it is small, always relevant and checked
when written, so it is added to every turn as its own block.

recall_memory needs only the question, so it can run beside document
retrieval; finalize_memory then packs its sentences against the document
context. The fused context is the document context plus memory_block().
Memory recall never breaks answering: any failure just means less memory.
"""

import logging
import math
import re

from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.api.dependencies.qdrant import qdrant_client
from app.core.config import settings
from app.services.context_monitor import CHARS_PER_TOKEN
from app.services.embedding_service import generate_query_embedding
from app.services.memory_state import handoff_block, state_block, current_state
from app.services.reconstruction import recall
from app.services.session_store import get_turns, user_key
from app.services.snapshot_store import list_snapshots


_log = logging.getLogger(__name__)


def memory_weight(turn: int) -> float:

    if settings.LAMBDA_FIXED is not None:
        return settings.LAMBDA_FIXED

    return 1 / (1 + math.exp(-settings.LAMBDA_BETA * (turn - settings.LAMBDA_THETA)))


def find_snapshots(user_id: str, chat_id: str, question: str) -> list[tuple[dict, float]]:
    """(snapshot, similarity) pairs of this chat, most similar first."""

    snapshots = {s["state_hash"]: s for s in list_snapshots(user_id, chat_id)}

    if not snapshots:
        return []

    hits = qdrant_client.query_points(
        collection_name=settings.SNAPSHOT_COLLECTION,
        query=generate_query_embedding(question),
        query_filter=Filter(must=[
            FieldCondition(key="user_key", match=MatchValue(value=user_key(user_id))),
            FieldCondition(key="chat_id", match=MatchValue(value=chat_id)),
        ]),
        limit=settings.SMS_TOP_K,
        score_threshold=settings.SMS_MIN_SIMILARITY,
    ).points

    return [
        (snapshots[hit.payload["state_hash"]], round(hit.score, 3))
        for hit in hits if hit.payload.get("state_hash") in snapshots
    ]


def merge_snapshots(snapshots: list[dict], active_document: str | None, with_conversation: bool = True) -> dict:
    """One snapshot-shaped view of several snapshots, for a single recall()
    call. With an active document, facts and entities tied to another
    document are left out; what the user said and decided belongs to the
    conversation and is kept, unless `with_conversation` is False (the
    conversation state block carries it instead)."""

    def same_document(documents) -> bool:
        return active_document is None or not documents or active_document in documents

    facets = {"facts": [], "user_facts": [], "entities": [], "decisions": [], "open_questions": [], "asked": []}

    for snapshot in sorted(snapshots, key=lambda s: s["from_turn"]):
        f = snapshot["facets"]
        facets["facts"] += [x for x in f["facts"] if same_document([x["document"]] if x.get("document") else [])]
        facets["entities"] += [x for x in f["entities"] if same_document(x.get("documents"))]
        facets["asked"] += f.get("asked", [])

        if with_conversation:
            facets["user_facts"] += f.get("user_facts", [])
            facets["decisions"] += f["decisions"]
            facets["open_questions"] += f["open_questions"]

    return {
        "state_hash": "+".join(s["state_hash"][:12] for s in snapshots),
        "from_turn": min(s["from_turn"] for s in snapshots),
        "to_turn": max(s["to_turn"] for s in snapshots),
        "facets": facets,
    }


def _normalize(text: str) -> str:

    return re.sub(r"\s+", " ", text.lower()).strip(" .")


def recall_memory(user_id: str, chat_id: str, question: str, active_document: str | None) -> dict | None:
    """Verified memory candidates for this turn, or None when memory is
    switched off. Call finalize_memory next."""

    if settings.MEMORY_RECALL_MODE == "off":
        return None

    turn = len(get_turns(user_id, chat_id)) + 1

    memory = {
        "turn": turn,
        "lambda": round(memory_weight(turn), 3),
        "snapshots": [],
        "sentences": [],
        "tokens": 0,
        "state_items": 0,
        "state_text": "",
        "handoff": False,
        "candidates": [],
    }

    if settings.MEMORY_STATE_ENABLED:
        try:
            memory["state_text"] = state_block(user_id, chat_id)
            memory["state_items"] = len(current_state(user_id, chat_id)) if memory["state_text"] else 0
            memory["handoff_text"] = handoff_block(user_id, chat_id)
            memory["handoff"] = bool(memory["handoff_text"])
        except Exception as exc:
            _log.warning("Conversation state unavailable for chat %s: %s", chat_id, exc)

    # "state": the conversation state only, no snapshot reconstruction.
    if settings.MEMORY_RECALL_MODE == "state":
        return memory

    try:

        found = find_snapshots(user_id, chat_id, question)

        if not found:
            return memory

        memory["snapshots"] = [{"seq": s["seq"], "similarity": score} for s, score in found]

        merged = merge_snapshots([s for s, _ in found], active_document, not settings.MEMORY_STATE_ENABLED)

        if not any(merged["facets"].values()):
            return memory

        result = recall(user_id, chat_id, merged, question, settings.MEMORY_RECALL_MODE)

    except Exception as exc:
        _log.warning("Memory recall failed for chat %s: %s", chat_id, exc)
        memory["error"] = str(exc)[:200]
        return memory

    memory["candidates"] = result["kept"]
    memory["dropped_unverified"] = len(result["dropped"])
    memory["verification_failed"] = result["verification_failed"]

    return memory


def finalize_memory(memory: dict | None, document_context: str) -> dict | None:
    """Pack the verified candidates into lambda(t) * MEMORY_CONTEXT_TOKENS,
    leaving out duplicates and what the document context already says."""

    if memory is None:
        return None

    budget = memory["lambda"] * settings.MEMORY_CONTEXT_TOKENS
    in_documents = _normalize(document_context)
    seen = set()

    memory["sentences"], memory["tokens"] = [], 0

    for item in memory.pop("candidates", []):

        key = _normalize(item["text"])
        tokens = max(1, len(item["text"]) // CHARS_PER_TOKEN)

        if key in seen or key in in_documents:
            continue

        if memory["tokens"] + tokens > budget:
            break

        seen.add(key)
        memory["sentences"].append(item)
        memory["tokens"] += tokens

    return memory


def memory_block(memory: dict | None) -> str:
    """The memory part of the fused context ("" when there is none)."""

    if not memory:
        return ""

    block = memory.get("handoff_text", "") + memory.get("state_text", "")

    if memory["sentences"]:
        lines = "\n".join(f"- {item['text']}" for item in memory["sentences"])
        block += f"\n\nVERIFIED MEMORY OF EARLIER TURNS (checked against the documents and the conversation):\n{lines}"

    return block
