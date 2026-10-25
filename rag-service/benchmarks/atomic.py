"""Result files are written whole or not at all: a run killed halfway through a write must not destroy hours of results."""

import json
import os


def dump_json(data, path: str) -> None:

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = path + ".tmp"

    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    os.replace(temporary, path)
