"""Test fixtures: no network, no models, no services.

    tmp_storage   chats, memory and registries in a temp folder
    llm           a scripted stand-in for the LLM gateway
    fake_embed    a bag-of-words embedding (similar texts score higher)
    qdrant        an in-memory stand-in for the snapshot index

Tests never touch the real storage/ folder, Qdrant, Redis or the gateway.
"""

import hashlib
import importlib
import os
import re
import sys

import numpy as np
import pytest

# Settings need these before app.core.config is imported.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-the-test-suite")
os.environ.setdefault("LLM_GATEWAY_TOKEN", "test-token")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings  # noqa: E402
from app.core.exceptions import LLMProviderError  # noqa: E402
from app.services import llm_provider  # noqa: E402


# ==========================================================
# Storage
# ==========================================================

@pytest.fixture(autouse=True)
def tmp_storage(tmp_path, monkeypatch):
    """Every test gets its own storage folder and the default settings."""

    monkeypatch.setattr(settings, "STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setattr(settings, "MEMORY_ROOT", None)
    monkeypatch.setattr(settings, "SNAPSHOT_COLLECTION", "snapshot_memory_test")

    from app.core import metrics
    metrics.reset()

    # The document pages a snapshot cites live in Qdrant; tests have none.
    from app.services import reconstruction

    monkeypatch.setattr(reconstruction, "get_page_texts", lambda document, pages: {})

    return tmp_path / "storage"


# ==========================================================
# LLM
# ==========================================================

class ScriptedLLM:
    """chat() replacement. `reply(match, text)` answers any prompt containing
    `match` (or for which `match(prompt)` is true); the first rule added that
    fits wins. An unmatched prompt behaves like an unreachable gateway."""

    def __init__(self):
        self.rules = []
        self.calls = []

    def reply(self, match, text):
        self.rules.append((match, text))
        return self

    def __call__(self, messages, stream=False, temperature=0, model=None, **kwargs):

        prompt = "\n".join(m["content"] for m in messages)
        self.calls.append(prompt)

        for match, text in self.rules:
            hit = match(prompt) if callable(match) else match in prompt
            if hit:
                content = text(prompt) if callable(text) else text
                if isinstance(content, Exception):
                    raise content
                return {"message": {"content": content}}

        raise LLMProviderError("no scripted reply for: " + prompt[:80])

    def calls_matching(self, text):
        return [c for c in self.calls if text in c]


@pytest.fixture
def llm(monkeypatch):
    """Replaces chat() everywhere it was imported by name."""

    fake = ScriptedLLM()
    real = llm_provider.chat

    # chat is imported by name, so every module that uses it must be loaded
    # before it can be patched.
    for name in (
        "context_monitor", "snapshot_engine", "reconstruction", "memory_state", "abrag",
        "verification_service", "generation_service", "followup_service", "hyde_service", "query_planner",
    ):
        importlib.import_module("app.services." + name)

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("app.") and getattr(module, "chat", None) is real:
            monkeypatch.setattr(module, "chat", fake)

    monkeypatch.setattr(llm_provider, "chat", fake)

    return fake


# ==========================================================
# Embeddings
# ==========================================================

DIM = 64


def embed_text(text: str) -> np.ndarray:

    vector = np.zeros(DIM)

    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0

    norm = np.linalg.norm(vector)

    return vector / norm if norm else vector


class FakeEmbeddingModel:

    def encode(self, texts, normalize_embeddings=True, **kwargs):

        if isinstance(texts, str):
            return embed_text(texts)

        return np.array([embed_text(t) for t in texts])


@pytest.fixture
def fake_embed(monkeypatch):

    model = FakeEmbeddingModel()

    from app.services import abrag, context_monitor, embedding_service, snapshot_engine

    for module in (abrag, context_monitor, embedding_service, snapshot_engine):
        monkeypatch.setattr(module, "get_embedding_model", lambda: model)

    # The cache in front of the query embedding needs Redis.
    monkeypatch.setattr(embedding_service, "get_embedding", lambda q: None)
    monkeypatch.setattr(embedding_service, "set_embedding", lambda q, e: None)

    return model


# ==========================================================
# Snapshot index (Qdrant)
# ==========================================================

class _Hit:

    def __init__(self, payload, score):
        self.payload, self.score = payload, score


class FakeQdrant:

    def __init__(self):
        self.collections = {}

    def get_collections(self):
        names = list(self.collections)
        return type("C", (), {"collections": [type("N", (), {"name": n}) for n in names]})()

    def create_collection(self, collection_name, vectors_config=None):
        self.collections.setdefault(collection_name, {})

    def upsert(self, collection_name, points):
        for point in points:
            self.collections[collection_name][point.id] = (np.array(point.vector), point.payload)

    def query_points(self, collection_name, query, query_filter=None, limit=10, score_threshold=None):

        query = np.array(query)
        hits = []

        for vector, payload in self.collections.get(collection_name, {}).values():

            if query_filter and not all(payload.get(c.key) == c.match.value for c in query_filter.must):
                continue

            score = float(vector @ query)

            if score_threshold is not None and score < score_threshold:
                continue

            hits.append(_Hit(payload, score))

        hits.sort(key=lambda h: h.score, reverse=True)

        return type("R", (), {"points": hits[:limit]})()


@pytest.fixture
def qdrant(monkeypatch):

    fake = FakeQdrant()

    from app.services import memory_retriever, snapshot_store

    monkeypatch.setattr(snapshot_store, "qdrant_client", fake)
    monkeypatch.setattr(memory_retriever, "qdrant_client", fake)

    return fake


# ==========================================================
# Chat helpers
# ==========================================================

@pytest.fixture
def chat_factory():
    """chat_factory(turns) -> (user_id, chat_id) with the (question, answer)
    pairs recorded as turns."""

    from app.services import session_store

    def make(turns, user_id="u1", **extra):
        chat_id = session_store.create_chat(user_id)
        for question, answer in turns:
            session_store.append_turn(user_id, chat_id, question, answer, **extra)
        return user_id, chat_id

    return make


def snapshot_json(facts=(), user_facts=(), decisions=(), open_questions=(), entities=(), asked=()):
    """What the snapshot extractor LLM would return."""

    import json

    return json.dumps({
        "facts": [{"text": t, "turn": n} for t, n in facts],
        "user_facts": [{"text": t, "turn": n} for t, n in user_facts],
        "entities": [{"name": name, "turns": turns} for name, turns in entities],
        "decisions": [{"topic": t, "status": s, "turn": n} for t, s, n in decisions],
        "open_questions": list(open_questions),
    })
