# SnapRAG — Build Plan & Research Paper Structure

> Goal: turn the current single-shot Qdrant+BM25 RAG service into the full
> SnapRAG system described in `a.txt`: **Self-Improving Snapshot Memory with
> Confidence-Gated Hybrid Retrieval for Long-Conversation Document QA**.

---

## PART 0 — Core Idea, Flow & Positioning (read this first)

> This part supersedes the broader framing below. The prior-art search (Oct 2026)
> showed most individual SnapRAG components already exist, so the paper is scoped
> around ONE defensible contribution: **verified reconstruction of lossy snapshot
> memory for long-conversation document QA.**

### 0.1 The idea in one paragraph

When a long conversation starts to overflow the context (or its quality degrades), the
old part is kept in two forms: the **full text in an immutable cold archive**, and a
**small lossy hot snapshot** (facts, numbers, dates, entities, and `doc/page/chunk`
anchors). When that old information is needed later, text is **reconstructed** from the
snapshot, and **every reconstructed sentence is verified** against document evidence and
the cold archive. Only verified content enters the prompt.

One line: *compress -> reconstruct later -> check against evidence -> only then use.*

### 0.2 Flow

```
Every turn
  |
  v
[1] TRIGGER      CIS (grounding, coherence, contradiction) or token pressure
  |               over threshold? -- no --> continue normally
  | yes
  v
[2] ARCHIVE      old full text -> cold bucket (as-is, never modified)
  |
  v
[3] ENCODE       text -> hot snapshot (lossy): facts + numbers + dates + entities
  |               + doc/page/chunk anchors   (~450 words -> ~50 tokens)
  v
---------------- later turns ----------------
  v
[4] RECALL       relevant snapshots, blended with document retrieval by
  |               lambda = sigmoid(beta * (turn - theta))
  v
[5] RECONSTRUCT  snapshot + document chunks -> LLM expands into full sentences
  |
  v
[6] VERIFY       each sentence checked against:
  |               - snapshot facts   - document chunks   - cold archive (if doubtful)
  |              unsupported -> drop or mark low-confidence
  v
[7] GENERATE     answer + citations from verified context;
                  low confidence -> retry (max 2)
```

Steps 5 and 6 are the research contribution. Steps 1, 2, 4 and 7 are supporting design.

### 0.3 What is novel and what is not

| Component | Already exists? | Novelty |
|---|---|---|
| Hybrid retrieval, RRF, rerank, HyDE | Yes, standard | None |
| Token-pressure compaction trigger | Yes (LCM, Deep Agents, Active Context Compression) | Low |
| Immutable raw archive + expand from summary | Yes (LCM) | Low |
| Versioned snapshots, rollback, fact supersede | Yes (ChronoMem, Cognis) | Low |
| Hash chain (`state_hash`/`prev_hash`) | Not in papers found; exists in OSS tools | Low, not a research claim |
| Keyword/entity lossy compression | Yes (MemCoRe, LLMLingua) | Low |
| Confidence-gated retry/corrective | Yes (CRAG, Self-RAG) | Low |
| Own prior snapshots as exemplars | Partly (Recursive Summarization uses labeled examples) | Some |
| CIS composite trigger | Not found | Good |
| Turn-aware lambda blend (document vs snapshot) | Not found | Good (small) |
| **Verify reconstruction against document evidence + archive** | LCM and MemCoRe do not describe it | **Best** |
| Long (100+ turn) document-QA conversations with page citations | MTRAG ~8 turns; LoCoMo is personal chat | Good (benchmark gap) |

Overall: moderate novelty. Enough for a strong workshop/short paper; a full conference
paper needs strong experiments. Caveat: this table rests on paper summaries and a limited
search, so repeat the search on Google Scholar / Semantic Scholar / arXiv before writing.

### 0.4 Scope decision

| Tier | Items |
|---|---|
| **Core (build and evaluate)** | Snapshot + cold archive, verified reconstruction (Phase 2b), CIS trigger (simple form), lambda blend, long-conversation doc-QA benchmark |
| **Supporting (build, but do not claim)** | Hash chain / versioning / immutability, confidence-gated retry, hybrid retrieval pipeline |
| **Defer / ablation only** | Handoff & session resume, exemplar self-improvement (claim only if ablation shows gain), multi-session memory |

### 0.5 Mandatory comparisons

- **LCM** (closest competitor: immutable store + token-threshold compaction, no verification).
- **Mem0** and **A-MEM** (memory baselines), **LLMLingua** (compression baseline).
- Full-context, summary-buffer, no-memory RAG.
- Position the claim honestly: near full-context accuracy at much lower tokens/latency,
  with lower hallucination from compressed memory. Do NOT claim "most accurate".

---

## PART 1 — Current State Analysis

### 1.1 What exists today (verified by reading the code)

**Stack**: FastAPI + uvicorn (port 8003), Qdrant (vector store), Redis (answer cache +
query-embedding cache), BM25Okapi (rank-bm25), sentence-transformers (BGE-small-en-v1.5,
384-dim), cross-encoder `ms-marco-MiniLM-L-6-v2`, PyMuPDF + Tesseract OCR for PDF
ingestion, python-jose JWT auth, OpenAI-compatible LLM gateway (`llm-gateway:8006/v1`,
Gemini flash-lite). Dockerfile boots CPU-only torch.

**Ingestion pipeline** (`upload.py` → `document_service.py`):
- Validates ext/MIME/size (50 MB cap), sanitizes filename, runs ingestion off the event
  loop (`run_in_threadpool`).
- Concurrency guard per filename (casefold) → per-owner overwrite rule → case-insensitive
  name-conflict rule → temp-file save → PDF text extraction with OCR fallback →
  parent/child chunking (3000/300 parent, 800/150 child) → embedding → **atomic index
  swap** (write new Qdrant vectors → delete old → commit staged BM25; rolled back on
  failure) → publish raw file via `os.replace` → register → clear Redis answer cache on
  re-upload.
- **Strong points**: full all-or-nothing semantics, rollback, per-user document registry,
  stale temp sweep on startup.

**Query pipeline** (`query_service.py` → `retrieval_pipeline.py`):
1. Redis answer-cache lookup (key `doc::question.lower()`).
2. Query understanding: classify (8 types) → per-type config table (`query_router.py`) →
   rewrite → decompose → self-query (metadata filters) → HyDE → dedupe.
3. Dense retrieval over Qdrant (one embedding per candidate query), active-document filter.
4. BM25 search → RRF fusion (k=60) → cross-encoder rerank → top-5 context.
5. Confidence (sigmoid over top-5 rerank scores) → LLM generation → followups → LLM
   verification (`verify_answer`) → if unsupported, answer replaced with a disclaimer.
6. `response_builder` returns sources/citations/context_chunks; SSE streaming variant.

**Engineering quality** (good):
- Observability: request-id propagation, `PerformanceLogger` stage timings, structured
  `rag_query` log event (`performance_logger.py`, `observability.py`).
- Security: JWT, filename sanitization, ownership checks; 20 test files incl. security,
  atomicity, and no-print tests. `conftest.py` pins Qdrant/Redis to 127.0.0.1:6399 in
  tests.

### 1.2 Gaps vs. the SnapRAG design (`a.txt`)

| Design stage | Status | Gap detail |
|---|---|---|
| Stage 1 Context Check (CQM) | **Missing** | No token-pressure / CIS / grounding-coherence-contradiction monitor. Nothing decides *when* to snapshot. |
| Stage 2 Query Understanding | Done | Covered by classifier/router/rewriter/decomposition/self-query/HyDE. |
| Stage 3A Document-store hybrid retrieval | Done | Qdrant dense + BM25 + RRF + rerank + parent/compression. |
| Stage 3B Memory Store (SMS-AR) | **Missing** | No snapshot vector store, no `λ=σ(β·(turn−θ))` blend, no context fusion A∪B with dedup. |
| Stage 4 Snapshot Engine | **Missing** | No facet extraction (facts/entities/decisions/open_questions), no `state_hash`, no versioned/immutable snapshots, no exemplar sampling. |
| Stage 5 Self-correcting generation | **Partial** | Verification + confidence exist but are single-pass. No `≥0.80 accept / 0.55–0.80 retry / ≤0.55 corrective` loop. |
| Stage 6 Output & Update | Partial | Response/citations/latency OK; no memory-update step (close open questions, add verified facts, supersede contradictions). |
| Handoff (session resume) | **Missing** | No session store, no handoff document injection. |
| Multi-turn continuity | **Broken** | `chat_memory.py` is a non-persistent module-global list; `add_message` is never called anywhere, so history is empty and not per-user/per-session. |

### 1.3 Other issues found

- `app/services/adaptive_retrievel.py` — dead code, misspelled filename, duplicates
  `query_router.py` with a different config shape; nothing imports it. **Delete or merge.**
- `process_query` and `process_query_stream` duplicate ~300 lines (retrieval/BM25/RRF
  steps). Extract a shared core.
- `query_service.py` builds the cache key inline with an f-string while
  `cache_service.make_cache_key` (sha256) is unused → inconsistent.
- ~8+ LLM calls per question (classify, rewrite, decompose, self-query, HyDE×k, generate,
  followups, verify) — needs gating/batching for free-tier rate limits (already noted in
  `.env.example`).
- No conversation/session endpoints, no conversation history persistence.
- No eval harness for long-conversation memory retention.
- No docker-compose in this repo (qdrant/redis/llm-gateway expected externally).

### 1.4 Cleanup and refactor already done (Oct 2026)

Resolved from 1.3: dead `adaptive_retrievel.py` deleted; docker-compose item dropped
(Docker removed entirely).

- **Removed:** `tests/`, `pytest.ini`, `requirements-dev.txt` (tests to be rebuilt later);
  `Dockerfile`, `.dockerignore`; `.env`, `.env.example`; `data/`; empty `app/middleware/`,
  `app/api/middleware/`, `alembic/`; unused modules `adaptive_retrievel.py`,
  `collection_service.py`, `query_services_classifier.py`, `context_expansion.py`,
  `parent_retriever.py`, `multi_query_service.py`; stray `app/services/bm25_indexes/`;
  duplicate PDFs.
- **`requirements.txt`** reduced from ~80 pinned lines to the direct dependencies only.
- **Observability replaced by LangSmith:** `core/observability.py` and
  `services/performance_logger.py` deleted. The OpenAI client is wrapped with
  `wrap_openai`, and pipeline functions carry `@traceable`. `core/logger.py` is now a
  plain `logging.basicConfig`. `main.py` calls `load_dotenv()` first so LangSmith sees
  its env vars.
- **To run again:** create a new `.env` with `JWT_SECRET_KEY`, `LLM_GATEWAY_TOKEN`,
  `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`; set
  `QDRANT_HOST`/`REDIS_HOST` to `localhost` (defaults are Docker hostnames) and
  `LLM_GATEWAY_URL`; install the Tesseract binary for OCR; recreate the venv.
- **Local test UI (Streamlit):** `frontend/app.py` (chat UI: streaming with live progress, batch mode with sources/confidence/follow-ups, document list + upload), `frontend/api.py` (backend client, no Streamlit imports), `frontend/.streamlit/config.toml` (light/dark theme). Run it from `frontend/` with `streamlit run app.py` (port 8501). `rag-service/dev_token.py` mints a dev JWT because the service has no login endpoint. Root `.gitignore` covers venv, `.env`, `rag-service/storage/`, caches, logs.
- **Robustness fixes (Oct 11):** `app/cache/redis_client.py` is one shared Redis client with a 30 s circuit breaker, so a Redis outage costs ~2 s once instead of failing (or stalling) every query; `/query/stream` now reports unexpected errors as an `error` event instead of dropping the connection; `pdf_extractor` finds Tesseract at its default Windows path and also OCRs screenshot-heavy pages (images >= 15% of the page and < 1000 chars of text), appending the OCR text to the caption.
- **Observed paper-relevant failure:** before the OCR fix, the PC0 screenshot was not indexed; the system answered with PC5's addresses yet reported confidence 0.99 and verification supported=True. Current verification checks support in the retrieved context, not entity consistency with the question. Keep this as a motivating example for verified reconstruction.
- **Lost with observability:** JSON logs, `X-Request-ID` header, `http_request` log line.
- **Streaming moved to LangGraph:** `api/routes/query_stream.py` and `process_query_stream` deleted. `services/query_graph.py` holds only the streaming (two nodes, retrieve -> generate); `POST /query/stream` lives in `routes/query.py` and only adds SSE framing. Event format (`step`/`token`/`error`/`done`) is unchanged. `langgraph==1.2.4` added to requirements.
- **Shared query core (Phase 0 item done):** `services/query_runner.py` (`run_retrieval`: understand -> dense -> BM25 -> RRF -> rerank, optional `on_step` callback) is used by both `process_query` and the stream. `query_service.py` shrank from ~930 to 95 lines.
- Note: LangSmith sends prompts, questions and retrieved chunks to a cloud service.

---

## PART 2 — Target Architecture

```
 Browser / API client
   │  (JWT, user_id)
   ▼
┌────────────────────── FastAPI  ──────────────────────┐
│  POST /query        (batched)                        │
│  POST /query/stream (SSE)                            │
│  POST /chats            → create session             │
│  GET  /chats/{id}       → resume (handoff)           │
│  POST /upload  · /documents · /documents/active      │
└────────────────────────┬─────────────────────────────┘
                         ▼
  ┌──────── STAGE 1 ──────────────┐
  │ Context Quality Monitor (CQM) │   n/a on turn 1
  │ CIS = w₁·grounding + w₂·tok(ρ) │
  │      + w₃·coherence − w₄·cont │
  │ tok_ρ < 0.20 or CIS < 0.55    │───────┐
  │   → snapshot trigger          │       │
  └──────────────┬───────────────┘       ▼
                 ▼               ┌─── STAGE 4 ──────────────┐
  ┌──────── STAGE 2 ─────────────┐│ Snapshot Engine          │
  │ classify → route → rewrite   ││ facets(LLM): facts,      │
  │ → decompose → self-query     ││  entities, decisions,    │
  │ → HyDE → dedupe              ││  open_questions          │
  └──────────────┬───────────────┘│ state_hash (versioned,   │
                 ▼                │  immutable)              │
  ┌──────── STAGE 3 ─────────────┐│ embed → snapshot store  │
  │ A. Document hybrid retrieval ││ sample 3-5 exemplars    │
  │    Qdrant dense + BM25       ││  for NEXT snapshot      │
  │    → RRF (k=60) → rerank     │└─────────────────────────┘
  │ B. Memory: snapshot recall   │
  │    top-k snapshots,          │
  │    λ = σ(β·(turn−θ))         │
  │ C. Fused context = A ∪ B     │
  │    (dedup, in-context size   │
  │     budget ∝ token pressure) │
  └──────────────┬───────────────┘
                 ▼
  ┌──────── STAGE 5 ──────────────┐
  │ Self-correcting generation    │
  │ LOOP max 2: generate→verify→  │
  │  confidence                   │
  │  ≥0.80 ACCEPT                 │
  │  0.55–0.80 RETRY (rewrite→    │
  │            re-retrieve→regen) │
  │  ≤0.55 CORRECTIVE (HyDE/exp)  │
  └──────────────┬───────────────┘
                 ▼
  ┌──────── STAGE 6 ──────────────┐
  │ Answer + citations, confidence│
  │ Memory UPDATE:                │
  │  close open_questions         │
  │  add verified facts           │
  │  supersede contradictions     │
  │ Performance/audit log         │
  └──────────────┬───────────────┘
                 ▼
         next turn → STAGE 1; session end → write Handoff doc
```

### 2.1 Data model

- **Session** (`sessions/{user_id}/{session_id}.json` or Redis): turns, `state_hash`, last
  snapshot id, token budget consumed, `token_pressure`, CQM history.
- **Snapshot** (`snapshots/{user_id}/{session_id}/{state_hash}.json`, immutable):
  ```
  {
    "state_hash":   "...sha256 of canonical facet json...",
    "created_at":   iso,
    "turn":         11,
    "prev_hash":    "<parent snapshot or null>",
    "facets": {
      "facts":         ["grounded, verifiable fact strings"],
      "entities":      [{"name", "document", "page", "aliases"}],
      "decisions":     [{"topic", "status": "closed|open", "turn"}],
      "open_questions":["strings", ...]
    },
    "exemplars_used":["<3-5 prior state_hashes>"],
    "embedding_v":   "<vector in snapshot store>",
    "meta": {"token_pressure", "cis", "trigger": "pressure|cis|manual"}
  }
  ```
- **Handoff doc** (`sessions/{user_id}/{session_id}/handoff.txt`, regenerated per
  snapshot/checkpoint): compressed summary injected into the prompt on resume.

### 2.2 New components to build

| Component | New files (proposed) |
|---|---|
| CQM (Stage 1) | `app/services/context_monitor.py` — measures token pressure `ρ = tokens_used/context_budget`, grounding/coherence/contradiction signals; computes CIS; decides trigger. |
| Snapshot Engine (Stage 4) | `app/services/snapshot_engine.py` (facet extraction prompt + JSON-schema validation, `state_hash`, versioning, immutability), `app/services/snapshot_store.py` (persistence). |
| Memory Store / SMS-AR (Stage 3B) | `app/services/memory_retriever.py` (snapshot top-k recall, `λ=σ(β·(turn−θ))`, fusion with doc context, dedup, context-budget packing). `app/repositories/snapshot_repository.py` (Qdrant collection `snapshot_memory` or Redis+FAISS). |
| Handoff (resume) | `app/services/handoff.py` — build/inject handoff doc at session start. |
| Self-correction loop (Stage 5) | `app/services/self_correct.py` — retry/corrective orchestration around existing `generate_answer/verify_answer/calculate_confidence`; threshold config. |
| Sessions & memory update (Stage 6) | `app/services/session_service.py`, `app/services/memory_updater.py` (close/supersede facts), routes `chats.py`. Persist via Redis (let it be the single source of truth for session/snapshot state). |
| Shared query core | Extract from `query_service.py` into `app/services/query_runner.py` used by both batch and stream paths. |

### 2.3 Config additions (`config.py` / `.env`)

```
CQM:  CQM_TRIGGER_PRESSURE=0.20, CQM_TRIGGER_CIS=0.55
      CQM_W_GROUNDING=0.35, CQM_W_TOKEN=0.25, CQM_W_COHERENCE=0.25, CQM_W_CONTRADICTION=0.15
SMS:  SMS_TOP_K=3, LAMBDA_BETA=0.08, LAMBDA_THETA=8
      SNAPSHOT_EMBED_DIM=384, SNAPSHOT_COLLECTION=snapshot_memory
GEN:  CONF_ACCEPT=0.80, CONF_RETRY=0.55, MAX_CORRECT_ATTEMPTS=2, ENABLE_CORRECTIVE=1
SESSION: SESSION_TTL_SECONDS, HANDOFF_MAX_TOKENS=600, CONTEXT_BUDGET_TOKENS=... (matches model window)
```

---

## PART 3 — Phased Build Plan (Do in this order)

### Phase 0 — Hardening & cleanup (0.5–1 day)
> Status: dead-code removal, Docker removal and LangSmith observability are DONE (see 1.4).
> Shared `query_runner` is also DONE. Still open: unified cache key (`cache_service.make_cache_key`), rebuilding tests.
- Delete `adaptive_retrievel.py`; delete or wire `chat_memory.py`.
- Unify cache key: use `cache_service.make_cache_key` everywhere.
- Extract shared retrieval core (`query_runner.py`) used by batch + streaming.
- Add `docker-compose.yml` (qdrant, redis, llm-gateway stub, rag-service) so the repo is
  self-contained; add `.dockerignore`.
- **Verify**: `pytest` still green.

### Phase 1 — Sessions & persistence (1–2 days)
- `POST /chats`, `GET /chats/{id}`, `GET /chats/{id}/messages`.
- Redis-backed conversation store (append-only turns per session; TTL).
- Replace `chat_memory` usage in `generation_prompt` with per-session history.
- **Tests**: session CRUD, history scoping, TTL, concurrency.

### Phase 2 — CQM + Snapshot Engine (2–3 days) — the research core
- `context_monitor.py`: compute `token_pressure ρ` per turn; grounding =
  verification-supported fraction; coherence = turn-over-turn embedding cosine;
  contradiction = deduped conflicting-fact detector (LLM/embedding); CIS formula; trigger
  rule from a.txt.
- `snapshot_engine.py`: facet extraction prompt (JSON-schema enforced via
  `extract_json`), canonicalize → `state_hash` = sha256(json canonical), write immutable
  snapshot file, store embedding in snapshot Qdrant collection, record exemplars used.
- `snapshot_store.py`: storage + retrieval + pruning (keep last N by session).
- Rerun extraction with 3–5 historical snapshots in prompt (self-improvement).
- **Tests**: trigger thresholds, hashing/versioning, immutability, facet schema, exemplar
  pass-through.

### Phase 2b — Snapshot codec + verified reconstruction (3–4 days) — THE paper contribution
- **Cold archive:** write the old turns verbatim, immutable, e.g.
  `archives/{user_id}/{session_id}/{turn_range}.jsonl.gz`.
- **Encoder (lossy):** extract atomic facts; force-keep numbers, dates and entities
  (compression studies show these are lost first); attach `doc/page/chunk_id` anchors.
  Aim for roughly 4x-10x compression (quality drops sharply beyond ~20x in LLMLingua and
  MemRefine).
- **Reconstruct:** snapshot + retrieved document chunks -> LLM expands to sentences.
  Test a stronger reconstruction model too (LLMLingua: GPT-4 recovers, GPT-3.5 does not).
- **Verify (sentence level):** support check against snapshot facts, then document chunks,
  then archive when doubtful (NLI or LLM judge). Unsupported sentences are dropped or
  marked low-confidence.
- **Metrics:** compression ratio, fact/entity recall of reconstruction, hallucination rate
  (unsupported facts / total facts), downstream QA accuracy.
- **Ablation ladder (paper's main figure):** keywords-only -> + document evidence -> + verify
  -> + archive fallback.

### Phase 3 — SMS-AR memory retrieval + fusion (2–3 days)
- `memory_retriever.py`: top-k snapshot recall (embedding + recency), `λ = σ(β·(turn−θ))`
  blend weight, fused context `A ∪ B` with dedup and token-budget packing (fill remaining
  context budget from snapshots when ρ is low, per design).
- Wire into shared query core; expose `retrieved_memory` in response.
- **Tests**: λ monotonic in turn, budget packing, dedup A∩B, active-doc isolation.

### Phase 4 — Self-correcting generation loop (1–2 days)
- `self_correct.py`: attempt 1 → verify → conf; `≥0.80 accept`; `0.55–0.80` → rewrite
  query + re-retrieve + regen (max 2 attempts); `≤0.55` → corrective (enable HyDE, expand
  top-k, optionally web tool) then regen. Always return best attempt + honest confidence.
- Replace the current "replace with ⚠️" fallback with this loop.
- **Tests**: mock LLM to hit each branch; assert attempt counts bounded.

### Phase 5 — Handoff & memory update (1–2 days)
- `handoff.py`: on session start, inject latest handoff doc (facts summary, open
  questions, entities) into the system prompt; on session end / snapshot trigger, write
  handoff doc with token budget.
- `memory_updater.py`: after each answer — move answered questions from
  `open_questions` → `decisions[closed]`, add newly verified facts (dedup by
  `state_hash`), supersede contradicted facts; update CQM metrics.
- **Tests**: resume continuity, fact supersession, open-question closure.

### Phase 6 — Eval harness + benchmarks (2–4 days, high-value for the paper)
- Build/extract a long-conversation document-QA eval set (see Part 4). Suggested: adapt
  **LoCoMo** (long-conversation memorization) or construct a synthetic dataset from your
  own PDF corpus (procedural multi-hop Qs + a 100-turn simulated conversation).
- Metrics: **turn-1 vs turn-100 answer accuracy** (memory retention), **RAGAS**-style
  faithfulness/answer-relevancy, **TOA** (throughout-answer correctness; Zhang et al.),
  **memory-loss = 1 − (late/early accuracy)**, fact-recall of snapshot facets, latency +
  LLM-call cost per turn.
- Baselines: current no-memory RAG, full-history-buffer RAG, MemGPT-style summarize-
  buffer, and SnapRAG (yours). Ablate: with/without CQM trigger, with/without SMS-AR λ,
  with/without exemplar self-improvement.
- Wire results into a `benchmarks/` folder + a run script (not a paper-only nicety — it
  is the paper's evidence).

### Phase 7 — Polish & productionization (1 day)
- Rate-limit middleware (LLM-call budget per user), optional request batching.
- OpenAPI tags + `/healthz` with dependency status (Qdrant/Redis).
- Metrics endpoint (Prometheus) reusing `PerformanceLogger`.
- Frontend (Next.js at :3000 per CORS config): chat sessions list, resume button,
  confidence + memory chip UI.

**Estimated total: ~2–3 focused weeks.**

---

## PART 4 — Research Paper Structure

> Working title: **SnapRAG: Verified Reconstruction of Snapshot Memory for
> Long-Conversation Document QA**

### 4.1 Abstract (4 sentences, fill numbers last)
1. Problem: in long conversations, earlier information is lost or, once compressed,
   reconstructed with hallucinations.
2. Gap: existing memory systems use compressed memory without checking it.
3. Method: lossy snapshots with an immutable cold archive; reconstruction verified against
   document evidence and the archive.
4. Result: hallucination reduced by X%, tokens reduced by Y% *(fill from Phase 6)*.

### 4.2 Contributions (three, honest)
1. **Verified reconstruction** of compressed snapshot memory against document evidence and
   an immutable archive.
2. A **quality-based trigger (CIS)** and a **turn-aware blend (lambda)** between document
   retrieval and snapshot memory.
3. A **long document-QA conversation benchmark** with ablations.

Do not list hash chain, versioning or immutability as contributions (LCM, ChronoMem and
Cognis cover them). Mention them in one line in Method.

### 4.3 Sections

| Section | Content | Show |
|---|---|---|
| 1 Introduction | Problem, gap, idea, contributions | Example: a fact from turn 5 is lost or distorted by turn 80 |
| 2 Related Work | (a) prompt compression: LLMLingua, ICAE; (b) conversation memory: Mem0, A-MEM, MemCoRe, LCM, ChronoMem, Cognis; (c) corrective RAG: CRAG, Self-RAG | Table: paper / what it does / difference |
| 3 Problem Setup | Notation: turn t, snapshot s, reconstruction s-hat, evidence E, hallucination rate | One formula for hallucination rate |
| 4 Method | 4.1 Trigger (CIS) · 4.2 Snapshot + archive · 4.3 Recall (lambda) · **4.4 Verified reconstruction** · 4.5 Confidence-gated generation | Figure 1 = flow from 0.2 |
| 5 Experimental Setup | Datasets: LoCoMo + own long doc-QA set; baselines per 0.5; metrics; seeds, bootstrap CI | Setup table |
| 6 Results | Compression ratio, fact recall, **hallucination rate**, accuracy turn 1 vs 100, latency, LLM calls | Main figure: ablation ladder |
| 7 Limitations | LLM cost, single LLM backbone, benchmark scope | |
| 8 Conclusion | One paragraph | |

### 4.4 Ablation table (evidence for each claim)

| Variant | Hallucination | Fact recall |
|---|---|---|
| Keywords-only reconstruction | ? | ? |
| + document evidence | ? | ? |
| + verify (full) | ? | ? |
| + archive fallback | ? | ? |
| CIS trigger vs fixed interval | ? | ? |
| Lambda fixed vs sigmoid | ? | ? |
| Exemplars on vs off | ? | ? |

### 4.5 Writing order
Experiments and results first, then Method, then Related Work, Introduction and Abstract
last. Leave abstract numbers blank until results exist.

### 4.6 Related-work reading list (read in full before citing; notes below came from summaries)

| Role | Paper | Why |
|---|---|---|
| Closest competitor / baseline | [LCM: Lossless Context Management](https://arxiv.org/html/2605.04050) | Immutable store + token-threshold compaction; no verification described |
| Core motivation | [Information Preservation in Prompt Compression](https://arxiv.org/html/2503.19114v2) | Reconstruction loses entities (28% kept) and grounding drops 30-50 points |
| Benchmark | [LoCoMo](https://arxiv.org/html/2402.17753v1) | ~305-turn conversations; observations beat summaries |
| Memory baseline | [Mem0](https://arxiv.org/html/2504.19413) | LoCoMo J-score, latency and token cost reference |
| Compression baseline | [LLMLingua](https://arxiv.org/html/2310.05736v2) | Token-level compression; recovery depends on model |
| Keyword tier | [MemCoRe](https://arxiv.org/html/2602.07885) | Notes/keywords/topics hierarchy; ~half of evidence unreachable |
| Also cite | [ChronoMem](https://arxiv.org/html/2607.27773), [Cognis](https://arxiv.org/html/2604.19771), [A-MEM](https://arxiv.org/html/2502.12110), [CRAG](https://arxiv.org/html/2401.15884v3), [ICAE](https://arxiv.org/html/2307.06945), [Recursive Summarization](https://arxiv.org/html/2308.15022v3), [Active Context Compression](https://arxiv.org/html/2601.07190), [Cost of Compression](https://arxiv.org/html/2609.12111), [MTRAG](https://arxiv.org/abs/2501.03468), [Rate-Distortion limits](https://arxiv.org/abs/2407.15504) | Related work and motivation |

Caveats: only a limited search was done; repeat it on Google Scholar, Semantic Scholar and
arXiv ("verified reconstruction", "memory hallucination", "context quality score") before
claiming novelty, and have a second person review the related-work table.

---

## Quick Start Checklist

- [ ] Phase 0 cleanup + shared `query_runner` + docker-compose
- [ ] Phase 1 sessions (Redis) + per-session history
- [ ] Phase 2 CQM + snapshot engine (immutable, versioned, exemplars)
- [ ] Phase 3 SMS-AR recall + λ-blend + fused context
- [ ] Phase 4 confidence-gated self-correction loop
- [ ] Phase 5 handoff injection + memory updater
- [ ] Phase 6 eval harness → numbers that fill the paper's Results

### Added after the prior-art review (Oct 2026)
- [ ] Phase 2b codec + cold archive + verified reconstruction (the paper contribution)
- [ ] Baselines: LCM, Mem0, A-MEM, LLMLingua, full-context, summary-buffer
- [ ] Own long document-QA conversation set (100+ turns, page citations)
- [ ] Ablation ladder: keywords-only -> + evidence -> + verify -> + archive
- [ ] Re-run literature search (Scholar / Semantic Scholar / arXiv) before claiming novelty
- [ ] Rebuild tests; recreate `.env`, venv, Tesseract; confirm LangSmith traces appear
