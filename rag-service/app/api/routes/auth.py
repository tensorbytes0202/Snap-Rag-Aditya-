import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import settings
from app.core.security import create_access_token, get_current_user_id
from app.services.rate_limiter import GUEST_PREFIX, start_guest_session, usage


router = APIRouter(tags=["auth"])


def _client_ip(request: Request) -> str:
    """The visitor's IP. The Streamlit frontend calls this API from its own
    server and forwards the viewer's IP in X-Forwarded-For. The header is
    easy to spoof, which is acceptable for a demo cap."""

    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()

    return forwarded or (request.client.host if request.client else "unknown")


@router.post("/auth/guest")
def guest_session(request: Request):
    """A limited demo token for a brand-new guest user."""

    if not settings.GUEST_ACCESS_ENABLED:
        raise HTTPException(status_code=403, detail="Guest access is turned off.")

    if not start_guest_session(_client_ip(request)):
        raise HTTPException(
            status_code=429,
            detail="Today's demo sessions for your network are used up. Please try again tomorrow.",
        )

    user_id = GUEST_PREFIX + uuid.uuid4().hex[:12]

    return {
        "token": create_access_token(user_id, settings.GUEST_TOKEN_HOURS),
        "user_id": user_id,
        "expires_in_hours": settings.GUEST_TOKEN_HOURS,
        "limits": usage(user_id),
    }


@router.get("/auth/usage")
def my_usage(user_id: str = Depends(get_current_user_id)):
    """The caller's remaining demo quota ({"guest": false} for full users)."""

    return usage(user_id)
