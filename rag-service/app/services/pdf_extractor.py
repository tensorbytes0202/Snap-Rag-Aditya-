import io
import logging
import os
import shutil

import fitz
import pytesseract
from PIL import Image


_log = logging.getLogger(__name__)

# Default install location of the UB Mannheim Windows build, used when
# Tesseract was installed but its folder is not on PATH (or PATH was
# updated after this process started).
_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_missing_tesseract_logged = False


def _tesseract_cmd() -> str | None:

    for candidate in (os.getenv("TESSERACT_CMD"), shutil.which("tesseract"), _WINDOWS_TESSERACT):
        if candidate and os.path.exists(candidate):
            return candidate

    return None

# A page with fewer extractable characters than this is treated as having
# no usable text layer (a scanned/image-only page) and falls back to OCR.
_OCR_MIN_TEXT_CHARS = 20

# Pages that are mostly screenshots with a short caption (e.g. "PC 0:" above
# a config window) also need OCR, otherwise the caption is indexed and the
# screenshot's content is lost. Long text pages with a figure are left alone,
# so textbooks don't get slow, noisy OCR on every page.
_OCR_IMAGE_COVERAGE = 0.15
_OCR_IMAGE_PAGE_MAX_TEXT_CHARS = 1000

# Tesseract's own recommended resolution: high enough for accuracy,
# without ballooning processing time on long documents.
_OCR_DPI = 200

def _ocr_page(page) -> str:
    """Rasterizes the page and runs it through Tesseract. Never raises --
    a missing Tesseract binary or an unreadable page degrades to an empty
    string rather than failing the whole upload."""

    global _missing_tesseract_logged

    cmd = _tesseract_cmd()

    if not cmd:
        if not _missing_tesseract_logged:
            _log.warning(
                "Tesseract not found, so image-only pages are indexed as blank. "
                "Install it (UB Mannheim build on Windows) or set TESSERACT_CMD, "
                "then re-upload the PDF."
            )
            _missing_tesseract_logged = True
        return ""

    pytesseract.pytesseract.tesseract_cmd = cmd

    try:
        pixmap = page.get_pixmap(dpi=_OCR_DPI)
        image = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return pytesseract.image_to_string(image)
    except Exception:
        _log.warning("OCR failed for page %s; leaving it blank.", page.number + 1, exc_info=True)
        return ""


def _image_coverage(page) -> float:
    """Fraction of the page area covered by embedded images."""

    area = abs(page.rect)
    if not area:
        return 0.0

    covered = sum(abs(fitz.Rect(info["bbox"]) & page.rect) for info in page.get_image_info())

    return covered / area


def _needs_visual(page, text_chars: int) -> bool:
    """Image-only pages, and mostly-screenshot pages with a short caption."""

    if text_chars < _OCR_MIN_TEXT_CHARS:
        return True

    return (
        text_chars < _OCR_IMAGE_PAGE_MAX_TEXT_CHARS
        and _image_coverage(page) >= _OCR_IMAGE_COVERAGE
    )


def extract_pdf_data(pdf_path):

    document = fitz.open(pdf_path)

    metadata = document.metadata

    texts = [document.load_page(n).get_text() for n in range(len(document))]

    visual_pages = [
        n for n in range(len(document))
        if _needs_visual(document.load_page(n), len(texts[n].strip()))
    ]

    pages = []

    for page_num in range(len(document)):

        text = texts[page_num]
        method = "text"

        if page_num in visual_pages:

            extra = _ocr_page(document.load_page(page_num))

            if extra.strip():
                method = "ocr"
                if len(text.strip()) < _OCR_MIN_TEXT_CHARS:
                    text = extra
                else:
                    # Keep the caption and add what the screenshots say.
                    text = text.rstrip() + "\n\n" + extra

        pages.append({"page_number": page_num + 1, "text": text, "method": method})

    _log.info("Extracted %d page(s): %d via OCR", len(pages), sum(p["method"] == "ocr" for p in pages))

    result = {
        "metadata": metadata,
        "total_pages": len(document),
        "pages": pages
    }

    document.close()

    return result
