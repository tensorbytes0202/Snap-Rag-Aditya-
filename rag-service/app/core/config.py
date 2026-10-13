import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ==========================================
    # Application
    # ==========================================

    APP_NAME: str = "Advanced RAG Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # ==========================================
    # Auth
    # ==========================================

    JWT_SECRET_KEY: str

    JWT_ALGORITHM: str = "HS256"

    # ==========================================
    # Guest (demo) access
    # ==========================================
    #
    # POST /auth/guest hands out a token for a fresh "guest-..." user so
    # reviewers can try the app without one. Guests are limited; tokens
    # minted with dev_token.py are not.

    GUEST_ACCESS_ENABLED: bool = True
    GUEST_TOKEN_HOURS: int = 24
    GUEST_QUERY_LIMIT: int = 7
    GUEST_UPLOAD_LIMIT: int = 2
    GUEST_MAX_PAGES: int = 40

    # Guest sessions one visitor (IP) may start per day, so reloading the
    # page doesn't reset the quota.
    GUEST_SESSIONS_PER_IP: int = 3

    # ==========================================
    # LLM Gateway
    # ==========================================

    LLM_GATEWAY_URL: str = "http://llm-gateway:8006/v1"

    LLM_GATEWAY_TOKEN: str

    LLM_MODEL: str = "gemini-3.5-flash-lite"

    # Reasoning effort sent to the gemini provider only (an empty
    # string means "don't send the parameter"). "minimal" is not
    # supported by gemini-3.8-flash -- use "low" or "" instead. The
    # gateway forwards unknown fields as-is to whichever provider a
    # request resolves to.
    LLM_REASONING_EFFORT: str = "low"

    # Per-call timeout for the gateway request. The gateway's own
    # worst case is its per-attempt timeout times its retry budget
    # (up to ~92s by default), so this must comfortably exceed that.
    LLM_REQUEST_TIMEOUT_SECONDS: float = 100.0

    # ==========================================
    # PDF page vision
    # ==========================================
    #
    # Image-heavy pages (screenshots, diagrams) are described by a
    # vision model instead of OCR, which loses which window/device a
    # value belongs to. OCR is still the fallback when this is off or
    # the gateway call fails. An empty VISION_MODEL means LLM_MODEL.

    PDF_VISION_ENABLED: bool = True

    VISION_MODEL: str = ""

    VISION_MAX_WORKERS: int = 4

    # ==========================================
    # Embeddings
    # ==========================================

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    CROSS_ENCODER_MODEL: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    EMBEDDING_DIM: int = 384

    # ==========================================
    # Qdrant
    # ==========================================

    QDRANT_HOST: str = "qdrant"

    QDRANT_PORT: int = 6333

    COLLECTION_NAME: str = "rag_documents"

    # ==========================================
    # Redis
    # ==========================================

    REDIS_HOST: str = "redis"

    REDIS_PORT: int = 6379

    REDIS_DB: int = 0

    CACHE_TTL: int = 3600

    # ==========================================
    # Chunking
    # ==========================================

    PARENT_CHUNK_SIZE: int = 3000

    PARENT_OVERLAP: int = 300

    CHILD_CHUNK_SIZE: int = 800

    CHILD_OVERLAP: int = 150

    # ==========================================
    # Storage
    # ==========================================
    #
    # Every consumer must read these via the
    # properties below (never cache them into a
    # module-level constant) so that STORAGE_DIR
    # can be swapped at runtime, e.g. in tests.

    STORAGE_DIR: str = "storage"

    @property
    def RAW_DATA_PATH(self) -> str:
        return os.path.join(self.STORAGE_DIR, "raw")

    @property
    def REGISTRY_FILE(self) -> str:
        return os.path.join(self.STORAGE_DIR, "document_registry.json")

    @property
    def ACTIVE_DOCUMENTS_FILE(self) -> str:
        return os.path.join(self.STORAGE_DIR, "active_documents.json")

    @property
    def BM25_INDEX_DIR(self) -> str:
        return os.path.join(self.STORAGE_DIR, "bm25_indexes")

    @property
    def CHATS_DIR(self) -> str:
        return os.path.join(self.STORAGE_DIR, "chats")

    @property
    def MEMORY_DIR(self) -> str:
        return os.path.join(self.STORAGE_DIR, "memory")

    @property
    def USAGE_FILE(self) -> str:
        return os.path.join(self.STORAGE_DIR, "usage.json")

    # ==========================================
    # Chat sessions
    # ==========================================

    # Most recent turns (question + answer) shown to the LLM so it can
    # resolve follow-ups like "what about Router-0?".
    HISTORY_TURNS: int = 3

    # ==========================================
    # Context Quality Monitor (CQM)
    # ==========================================
    #
    # CIS = w_g*grounding + w_t*headroom + w_c*coherence - w_x*contradiction,
    # measured over the turns not yet in a snapshot. A snapshot is taken
    # when headroom < CQM_TRIGGER_HEADROOM or CIS < CQM_TRIGGER_CIS.
    # Headroom = 1 - tokens of those turns / MEMORY_BUDGET_TOKENS.

    CQM_W_GROUNDING: float = 0.35
    CQM_W_TOKEN: float = 0.25
    CQM_W_COHERENCE: float = 0.25
    CQM_W_CONTRADICTION: float = 0.15

    CQM_TRIGGER_HEADROOM: float = 0.20
    CQM_TRIGGER_CIS: float = 0.55

    # Budget for raw (not yet snapshotted) conversation; ~10 typical
    # turns fill 80% of it.
    MEMORY_BUDGET_TOKENS: int = 2000

    # ==========================================
    # Snapshot engine
    # ==========================================

    # Previous snapshots of the chat shown to the extractor as examples
    # (self-improvement); 0 turns it off for ablations.
    SNAPSHOT_EXEMPLARS: int = 3

    SNAPSHOT_COLLECTION: str = "snapshot_memory"

    # ==========================================
    # Verified reconstruction
    # ==========================================

    # Model that rebuilds and verifies memory; empty means LLM_MODEL.
    RECONSTRUCTION_MODEL: str = ""

    # Cap on document text (from the snapshot's page anchors) sent along
    # with a snapshot; ~3k tokens.
    EVIDENCE_MAX_CHARS: int = 12000

    # ==========================================
    # Memory recall during answering (SMS-AR)
    # ==========================================
    #
    # lambda(t) = sigmoid(LAMBDA_BETA * (t - LAMBDA_THETA)), t = turn number.
    # Verified memory may use up to lambda(t) * MEMORY_CONTEXT_TOKENS of the
    # prompt; the document context is not reduced.

    LAMBDA_BETA: float = 0.08
    LAMBDA_THETA: float = 8

    # Ablation: a constant lambda instead of the sigmoid.
    LAMBDA_FIXED: float | None = None

    MEMORY_CONTEXT_TOKENS: int = 600

    # Snapshots recalled per question, and the minimum question/snapshot
    # cosine similarity for a snapshot to be considered at all.
    SMS_TOP_K: int = 3
    SMS_MIN_SIMILARITY: float = 0.5

    # "off" (no memory: the baseline) or a reconstruction mode
    # ("snapshot", "evidence", "verify", "archive" = full method).
    MEMORY_RECALL_MODE: str = "archive"

    # ==========================================
    # Pydantic Settings
    # ==========================================

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
