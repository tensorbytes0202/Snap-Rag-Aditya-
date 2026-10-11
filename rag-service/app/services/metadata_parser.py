import os
import re

def extract_metadata(question: str):

    metadata = {
        "page": None,
        "document_name": None
    }

    question_lower = question.lower()

    page_patterns = [
        r"page\s+(\d+)",
        r"pg\s+(\d+)",
        r"page\s*:\s*(\d+)",
        r"page#(\d+)"
    ]

    for pattern in page_patterns:
        match = re.search(pattern, question_lower)

        if match:
            metadata["page"] = int(match.group(1))
            break

    pdf_match = re.search(
        r'([A-Za-z0-9_\- ]+\.pdf)',
        question,
        re.IGNORECASE
    )

    if pdf_match:
        metadata["document_name"] = os.path.basename(pdf_match.group(1)).strip()

    return metadata