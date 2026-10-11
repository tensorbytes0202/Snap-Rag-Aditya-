import base64
import logging

from app.core.config import settings
from app.prompts.page_vision_prompt import build_page_vision_prompt
from app.services.llm_provider import chat


_log = logging.getLogger(__name__)


def describe_page(image_jpeg: bytes, page_text: str, previous_text: str) -> str:
    """Asks the vision model to transcribe a rendered page. Never raises --
    a gateway/model failure returns "" so the caller can fall back to OCR."""

    image_url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_page_vision_prompt(page_text, previous_text)},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]

    try:
        response = chat(messages, model=settings.VISION_MODEL or None)
        return response["message"]["content"] or ""
    except Exception:
        _log.warning("Vision description failed; falling back to OCR.", exc_info=True)
        return ""
