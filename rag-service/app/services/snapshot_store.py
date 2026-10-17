"""Memory files of a chat, and the snapshot search index.

storage/memory/<user key>/<chat_id>/
    cqm.jsonl                      one line per measured turn (CIS + signals)
    snapshot_001_<hash12>.json     immutable snapshots, in order

The snapshot files are the source of truth. The Qdrant collection
SNAPSHOT_COLLECTION is a search index built from them, so a failed index
write never loses a snapshot.
"""

import glob
import json
import logging
import os
import uuid

from qdrant_client.models import Distance, PointStruct, VectorParams

from app.api.dependencies.qdrant import qdrant_client
from app.core.config import settings
from app.services.session_store import append_jsonl, is_chat_id, read_jsonl, user_key


_log = logging.getLogger(__name__)


def _chat_dir(user_id: str, chat_id: str) -> str:

    if not is_chat_id(chat_id):
        raise ValueError("malformed chat id")

    return os.path.join(settings.MEMORY_DIR, user_key(user_id), chat_id)


# ==========================================================
# Snapshots
# ==========================================================

def list_snapshots(user_id: str, chat_id: str) -> list[dict]:
    """The chat's snapshots, oldest first."""

    paths = sorted(glob.glob(os.path.join(_chat_dir(user_id, chat_id), "snapshot_*.json")))

    snapshots = []

    for path in paths:
        with open(path, encoding="utf-8") as f:
            snapshots.append(json.load(f))

    return snapshots


def save_snapshot(user_id: str, chat_id: str, snapshot: dict) -> None:
    """Write a snapshot once; an existing file is never overwritten."""

    folder = _chat_dir(user_id, chat_id)
    os.makedirs(folder, exist_ok=True)

    name = f"snapshot_{snapshot['seq']:03d}_{snapshot['state_hash'][:12]}.json"

    with open(os.path.join(folder, name), "x", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def index_snapshot(user_id: str, chat_id: str, snapshot: dict, text: str, vector: list[float]) -> None:
    """Add a snapshot to the search index (one point per snapshot)."""

    collection = settings.SNAPSHOT_COLLECTION

    if collection not in [c.name for c in qdrant_client.get_collections().collections]:
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
        )

    qdrant_client.upsert(
        collection_name=collection,
        points=[PointStruct(
            id=str(uuid.UUID(snapshot["state_hash"][:32])),
            vector=vector,
            payload={
                "user_key": user_key(user_id),
                "chat_id": chat_id,
                "seq": snapshot["seq"],
                "state_hash": snapshot["state_hash"],
                "from_turn": snapshot["from_turn"],
                "to_turn": snapshot["to_turn"],
                "text": text,
            },
        )],
    )


# ==========================================================
# CQM log
# ==========================================================

def read_cqm(user_id: str, chat_id: str) -> list[dict]:

    return read_jsonl(os.path.join(_chat_dir(user_id, chat_id), "cqm.jsonl"))


def append_cqm(user_id: str, chat_id: str, record: dict) -> None:

    folder = _chat_dir(user_id, chat_id)
    os.makedirs(folder, exist_ok=True)

    append_jsonl(os.path.join(folder, "cqm.jsonl"), record)


# ==========================================================
# Conversation state versions
# ==========================================================

def read_state_versions(user_id: str, chat_id: str) -> list[dict]:
    """Every saved version of the conversation state, oldest first."""

    return read_jsonl(os.path.join(_chat_dir(user_id, chat_id), "state.jsonl"))


def append_state_version(user_id: str, chat_id: str, record: dict) -> None:
    """Versions are appended, never rewritten: the state has a history."""

    folder = _chat_dir(user_id, chat_id)
    os.makedirs(folder, exist_ok=True)

    append_jsonl(os.path.join(folder, "state.jsonl"), record)
