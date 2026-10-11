import json


def extract_json(text: str):
    """
    Parse a JSON object/array out of an LLM response.

    Strips a surrounding ``` or ```json fence (and whitespace) before
    attempting to parse. Returns None if the text cannot be parsed as
    JSON, instead of raising.
    """

    if not text:
        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):

        newline_index = cleaned.find("\n")

        if newline_index != -1:
            cleaned = cleaned[newline_index + 1:]
        else:
            cleaned = cleaned[3:]

        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None
