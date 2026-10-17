from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_current_user_id
from app.services.memory_state import build_handoff, current_state
from app.services.session_store import chat_exists, create_chat, get_turns, list_chats
from app.services.snapshot_store import list_snapshots, read_cqm, read_state_versions


router = APIRouter(tags=["chats"])


def _require_chat(user_id: str, chat_id: str) -> None:
    # Another user's chat is reported as missing, not forbidden, so chat ids
    # can't be probed.
    if not chat_exists(user_id, chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")


@router.post("/chats")
def new_chat(user_id: str = Depends(get_current_user_id)):

    return {"chat_id": create_chat(user_id)}


@router.get("/chats")
def chats(user_id: str = Depends(get_current_user_id)):

    return {"chats": list_chats(user_id)}


@router.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: str, user_id: str = Depends(get_current_user_id)):

    _require_chat(user_id, chat_id)

    return {"chat_id": chat_id, "turns": get_turns(user_id, chat_id)}


@router.get("/chats/{chat_id}/memory")
def chat_memory(chat_id: str, user_id: str = Depends(get_current_user_id)):
    """The context-monitor log (one entry per turn), the snapshots and the
    conversation state with its history of versions."""

    _require_chat(user_id, chat_id)

    return {
        "chat_id": chat_id,
        "cqm": read_cqm(user_id, chat_id),
        "snapshots": list_snapshots(user_id, chat_id),
        "state": current_state(user_id, chat_id),
        "state_versions": len(read_state_versions(user_id, chat_id)),
    }


@router.get("/chats/{chat_id}/handoff")
def chat_handoff(chat_id: str, user_id: str = Depends(get_current_user_id)):
    """Where the conversation stands: the document to resume a chat from."""

    _require_chat(user_id, chat_id)

    return {
        "chat_id": chat_id,
        "handoff": build_handoff(user_id, chat_id),
        "state": current_state(user_id, chat_id),
    }
