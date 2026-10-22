"""Checks dataset.json against the PDF: every fact's evidence pattern must occur
in the gold pages of its question. Run after editing the dataset.

    python -m benchmarks.docqa.validate
"""

import json
import os
import re
import sys

import fitz

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def page_texts(pdf_path: str) -> list[str]:
    """Page text with line breaks and odd spacing flattened, as patterns expect."""

    with fitz.open(pdf_path) as document:
        return [re.sub(r"\s+", " ", page.get_text().replace("\ufffd", "'")) for page in document]


def main() -> int:

    data = json.load(open(os.path.join(ROOT, "benchmarks", "docqa", "dataset.json"), encoding="utf-8"))
    pages = page_texts(os.path.join(ROOT, "storage", "raw", data["document"]))
    problems = 0

    for item in data["questions"]:

        gold = " ".join(pages[p - 1] for p in item["pages"])

        for fact in item["facts"]:

            pattern = fact.get("e", fact["a"])
            where = [p for p in item["pages"] if re.search(pattern, pages[p - 1], re.I)]

            if not where:
                problems += 1
                print(f"{item['id']}: evidence {pattern!r} not found on pages {item['pages']}")

        if item.get("answerable", True) and len(gold) < 200:
            problems += 1
            print(f"{item['id']}: gold pages {item['pages']} have almost no text")

    print(f"{len(data['questions'])} questions checked, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
