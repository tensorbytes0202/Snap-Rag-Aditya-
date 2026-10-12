"""Chat sessions: one append-only JSONL file per chat.

storage/chats/<user key>/<chat_id>.jsonl holds one line per turn (a question
and its answer). Lines are only ever appended, never rewritten, so the file is
the full, unmodified record of the conversation: later phases compress old
turns into snapshots and verify reconstructions against this record.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid

from app.core.config import settings


_log = logging.getLogger(__name__)

_lock = threading.Lock()

_CHAT_ID = re.compile(r"[0-9a-f]{32}")


def _user_dir(user_id: str) -> str:

    # The user id comes from the JWT; hashed so it is always a safe folder name.
    return os.path.join(settings.CHATS_DIR, hashlib.sha256(user_id.encode()).hexdigest()[:16])


def _chat_path(user_id: str, chat_id: str) -> str | None:
    """None for a malformed chat id, so a request can never reach a path
    outside the user's own folder."""

    if not isinstance(chat_id, str) or not _CHAT_ID.fullmatch(chat_id):
        return None

    return os.path.join(_user_dir(user_id), chat_id + ".jsonl")


def create_chat(user_id: str) -> str:

    chat_id = uuid.uuid4().hex
    path = _chat_path(user_id, chat_id)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "x", encoding="utf-8").close()

    return chat_id


def chat_exists(user_id: str, chat_id: str) -> bool:

    path = _chat_path(user_id, chat_id)

    return path is not None and os.path.isfile(path)


def get_turns(user_id: str, chat_id: str) -> list[dict]:

    path = _chat_path(user_id, chat_id)

    if path is None or not os.path.isfile(path):
        return []

    turns = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                turns.append(json.loads(line))
            except ValueError:
                # Only a crash mid-write can leave a partial line; skip it
                # rather than make the whole chat unreadable.
                _log.warning("Skipping unreadable line in chat %s", chat_id)

    return turns


def append_turn(user_id: str, chat_id: str, question: str, answer: str, **extra) -> dict:
    """Record one question/answer turn; `extra` holds metadata such as the
    document, citations and verification result."""

    path = _chat_path(user_id, chat_id)

    with _lock:

        turn = {
            "turn": len(get_turns(user_id, chat_id)) + 1,
            "ts": round(time.time(), 3),
            "question": question,
            "answer": answer,
            **extra,
        }

        # Start on a fresh line even if a crash left a partial one behind,
        # otherwise this turn would be glued onto it and lost too.
        prefix = "" if _ends_with_newline(path) else "\n"

        with open(path, "a", encoding="utf-8") as f:
            f.write(prefix + json.dumps(turn, ensure_ascii=False) + "\n")

    return turn


def _ends_with_newline(path: str) -> bool:

    with open(path, "rb") as f:

        if f.seek(0, os.SEEK_END) == 0:
            return True

        f.seek(-1, os.SEEK_END)

        return f.read(1) == b"\n"


def list_chats(user_id: str) -> list[dict]:
    """The user's chats, most recently updated first."""

    folder = _user_dir(user_id)

    if not os.path.isdir(folder):
        return []

    chats = []

    for name in os.listdir(folder):

        chat_id = name.removesuffix(".jsonl")

        if not name.endswith(".jsonl") or not _CHAT_ID.fullmatch(chat_id):
            continue

        turns = get_turns(user_id, chat_id)

        chats.append({
            "chat_id": chat_id,
            "title": turns[0]["question"][:80] if turns else "New chat",
            "turns": len(turns),
            "updated_at": os.path.getmtime(os.path.join(folder, name)),
        })

    return sorted(chats, key=lambda chat: chat["updated_at"], reverse=True)
