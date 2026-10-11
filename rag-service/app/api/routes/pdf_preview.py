import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.security import get_current_user_id
from app.services.pdf_service import validate_stored_filename


logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/pdf/{filename}")
def get_pdf(
    filename: str,
    user_id: str = Depends(get_current_user_id),
):
    # -------------------------------------------------------
    # Path Traversal Protection
    # Reduce the requested name to a safe basename first.
    # -------------------------------------------------------
    try:
        safe_name = validate_stored_filename(filename)
    except ValueError:
        logger.warning("Blocked path traversal attempt: %s", filename)
        raise HTTPException(status_code=400, detail="Invalid filename.")

    upload_dir = Path(settings.RAW_DATA_PATH).resolve()
    pdf_path = (upload_dir / safe_name).resolve()

    # Double-check the resolved path is still inside upload_dir
    if pdf_path != upload_dir and upload_dir not in pdf_path.parents:
        logger.warning("Resolved path escapes upload dir: %s", pdf_path)
        raise HTTPException(status_code=400, detail="Invalid filename.")

    logger.debug("Serving PDF: %s", pdf_path)

    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found.")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
    )
