"""Thin client for the SnapRAG backend. No Streamlit imports, so it is easy to test."""

import json
from collections.abc import Iterable, Iterator

import requests


class ApiError(Exception):
    """A readable error from the backend or the network."""


def ping(base_url: str) -> bool:
    try:
        return requests.get(base_url.strip().rstrip("/") + "/", timeout=2).ok
    except requests.RequestException:
        return False


def _describe(resp: requests.Response) -> str:
    try:
        detail = resp.json().get("detail", resp.reason)
    except (ValueError, AttributeError):
        detail = resp.reason or "request failed"

    message = f"{resp.status_code}: {detail}"

    if resp.status_code in (401, 403):
        message += " (check the access token; `python dev_token.py` makes one)"

    return message


def parse_events(chunks: Iterable[bytes]) -> Iterator[dict]:
    """Turn raw SSE bytes into event dicts. Buffers bytes (not text), so a
    multi-byte character split across two chunks is never corrupted."""

    buffer = b""

    for chunk in chunks:
        buffer += chunk

        while b"\n\n" in buffer:
            block, buffer = buffer.split(b"\n\n", 1)
            data = b"\n".join(
                line[6:] for line in block.split(b"\n") if line.startswith(b"data: ")
            )
            if data:
                yield json.loads(data.decode("utf-8"))


class SnapRagClient:

    def __init__(self, base_url: str, token: str):
        self.base = base_url.strip().rstrip("/")
        token = token.strip()
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def _request(self, method: str, path: str, timeout, **kwargs) -> requests.Response:
        try:
            resp = requests.request(
                method, self.base + path, headers=self.headers, timeout=timeout, **kwargs
            )
        except requests.RequestException as exc:
            raise ApiError(f"Cannot reach {self.base} ({exc.__class__.__name__})") from exc

        if not resp.ok:
            raise ApiError(_describe(resp))

        return resp

    def documents(self) -> dict:
        return self._request("GET", "/documents", timeout=10).json()

    def select_document(self, name: str) -> None:
        self._request("POST", "/documents/active/" + requests.utils.quote(name, safe=""), timeout=10)

    def upload(self, filename: str, data: bytes) -> dict:
        files = {"file": (filename, data, "application/pdf")}
        return self._request("POST", "/upload", timeout=(5, 900), files=files).json()

    def create_chat(self) -> str:
        return self._request("POST", "/chats", timeout=10).json()["chat_id"]

    def query(self, question: str, chat_id: str | None = None) -> dict:
        body = {"question": question, "chat_id": chat_id}
        return self._request("POST", "/query", timeout=(5, 180), json=body).json()

    def stream(self, question: str, chat_id: str | None = None) -> Iterator[dict]:
        connected = False

        try:
            with requests.post(
                self.base + "/query/stream",
                json={"question": question, "chat_id": chat_id},
                headers=self.headers,
                stream=True,
                timeout=(5, 120),
            ) as resp:
                connected = True

                if not resp.ok:
                    raise ApiError(_describe(resp))

                yield from parse_events(resp.iter_content(chunk_size=None))

        except requests.RequestException as exc:
            problem = "Connection lost" if connected else f"Cannot reach {self.base}"
            raise ApiError(f"{problem} ({exc.__class__.__name__})") from exc
