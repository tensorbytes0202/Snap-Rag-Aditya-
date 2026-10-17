# SnapRAG

**Verified Reconstruction of Snapshot Memory for Long-Conversation Document QA**

![CI](https://github.com/tensorbytes0202/Snap-Rag-Aditya-/actions/workflows/ci.yml/badge.svg)

SnapRAG is a retrieval-augmented question-answering system for long conversations about
PDF documents. When a conversation grows, older turns are compressed into small
*snapshots*. When that older information is needed again, SnapRAG rebuilds it from the
snapshot and **checks every rebuilt sentence against evidence** (the snapshot itself, the
document pages it points to, and the untouched original conversation). Only verified
memory is allowed into the prompt.

> compress → reconstruct later → verify against evidence → only then use

This repository holds both the working system and the research prototype behind a paper
in progress (field: NLP / information retrieval; RAG with long-term conversational memory).

---

## Contents

- [Why](#why)
- [How it works](#how-it-works)
- [What is novel](#what-is-novel)
- [Project status](#project-status)
- [Preliminary results](#preliminary-results)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Running](#running)
- [API](#api)
- [Research settings](#research-settings)
- [Benchmark](#benchmark)
- [Data and privacy](#data-and-privacy)

---

## Why

In long conversations, early information is either **lost** (pushed out of the context
window) or **distorted** once compressed: summaries drop numbers and entity names, and
expanding them again invites hallucination. Existing conversation-memory systems put
compressed memory straight back into the prompt without checking it.

SnapRAG keeps old turns in two forms:

| Form | What it holds | Properties |
|---|---|---|
| **Cold archive** | every turn, verbatim | append-only, never rewritten |
| **Hot snapshot** | facts, entities, decisions, open questions, with document/page anchors | small, lossy, immutable, hash-chained |

---

## How it works

```
Every question
  ├─ side by side
  │    Document retrieval   query plan → dense (Qdrant) + BM25 → RRF → cross-encoder rerank
  │                         → full parent pages of the top hits + a one-line-per-page overview
  │    Memory recall        (inside a chat) conversation state + similar snapshots → reconstruct
  │                         → verify → keep verified sentences, within λ(t) × 600 tokens
  ├─ Answer                 generated from documents + memory (+ the turns not yet in a snapshot)
  ├─ Check (AB-RAG)         answer verification → confidence from three signals
  │    ≥ 0.80 return · below it read the next batch of passages while the intent's budget lasts
  │    the best attempt is returned with its honest confidence
  └─ Record                 turn appended to the chat log (the cold archive)

After every turn (background task)
  ├─ Context Quality Monitor
  │     CIS = 0.35·grounding + 0.25·headroom + 0.25·coherence − 0.15·contradiction
  │     take a snapshot if headroom < 0.20 or CIS < 0.55
  ├─ Snapshot engine
  │     LLM extracts facts / user statements / entities / decisions / open questions from the
  │     un-snapshotted turns; code adds "asked at turn N" and the page anchors
  │     → state_hash (sha256), chained via prev_hash → written once + indexed in Qdrant
  └─ Memory updater
        decisions, user statements and open questions of every new snapshot are checked against
        the user's own messages, consolidated (a new value replaces the old one, an answered
        question closes) and saved as a new version of the conversation state + a hand-off document
```

### Adaptive budgeted retrieval (AB-RAG)

A fixed top-k wastes work on easy questions and starves hard ones. Following the AB-RAG idea
(adaptive retrieval effort under a budget, see the paper in `SNAPRAG_PLAN.md`), the first batch
of passages is answered and scored, and more passages are read only while the score is below
the threshold and budget remains:

```
confidence = 0.50·S1 + 0.25·S2 + 0.25·S3
  S1  model certainty         the checker's confidence (no token probabilities from the gateway: a proxy)
  S2  answer-evidence match   cosine of the answer to its closest evidence text
  S3  reranker-score spread   std of the reranker scores of the passages read
```

An answer the checker does not support is capped below the threshold. What this adds: the first
batch and the budget follow the **intent** of the question (a definition gets 3 passages and 1 extra
batch, a summary 8 and 3); the evidence pool includes the **verified conversation memory**, and a
question about the chat itself never goes back to the documents; the next batch is free (the same
ranking read deeper) until the ranking is used up, and only then a fresh, wider search runs, aimed
by what the checker could not find. Weights and threshold are uncalibrated defaults; the benchmark's
`full_fixed_k` variant (no extra batches) is the fixed-k ablation.

### Conversation state and hand-off

What the user says about their own task ("my deadline is 15 November", "skip sections 32 and 33")
is in no document and tiny, so it does not compete with document facts for a snapshot slot. It is
kept as a consolidated **conversation state**: every item was checked against the user's own
messages when written, the LLM only points at which old items a new one replaces (it never
rewrites text), and every update appends a version to `state.jsonl`. The state is shown on every
turn. A chat idle for more than 30 minutes is a resumed session: its first turn also gets the
**hand-off** document (state, last question, recent topics, key entities), which
`GET /chats/{id}/handoff` returns too.

### Verified reconstruction (the core method)

```
snapshot + the document pages it is anchored to
   └─► LLM rebuilds short, self-contained sentences
          └─► each sentence is checked, in order:
                 1. stated by the snapshot facts or the anchored pages?   → keep
                 2. otherwise stated by the cold archive (original turns)? → keep
                 3. otherwise                                             → drop
```

If the checking step itself fails, nothing counts as verified.

### Memory weight

```
λ(t) = sigmoid(β · (t − θ)),  β = 0.08, θ = 8     (t = turn number)
λ(1) = 0.36   λ(8) = 0.50   λ(20) = 0.72   λ(50) = 0.97
```

λ controls how much verified memory may enter the prompt. The document context is never
reduced.

### Indexing

PDF text is extracted with PyMuPDF. Image-heavy pages (screenshots, diagrams) are read by a
vision model, so values stay attached to the right device or window; Tesseract OCR is the
fallback. Documents are split into parent (3000 chars) and child (800 chars) chunks,
embedded with `BAAI/bge-small-en-v1.5`, and stored in Qdrant and a BM25 index.

---

## What is novel

| Contribution | Status |
|---|---|
| **C1. Verified reconstruction** of compressed memory against snapshot, document evidence and an immutable archive | implemented |
| **C2. Quality-based snapshot trigger (CIS)** and a turn-aware memory weight (λ) | implemented |
| **C4. Intent- and memory-aware adaptive budgeted retrieval** (AB-RAG signals, intent budgets, verified memory as evidence) | implemented; unvalidated |
| **C3. Long document-QA conversation benchmark** (100 turns, page citations) | harness, dataset spec and scoring done; results pending |

Hybrid retrieval, hash chains and immutability are supporting design, and AB-RAG's confidence-gated
stopping is the cited prior work that C4 builds on, not a claimed contribution (see also LCM,
ChronoMem, Cognis, CRAG, Self-RAG). None of the novelty claims has had a full literature check yet. The full plan,
related work and paper outline are in [`SNAPRAG_PLAN.md`](SNAPRAG_PLAN.md).

---

## Project status

| Phase | Work | Status |
|---|---|---|
| 0 | Cleanup, shared retrieval core, LangSmith tracing | done |
| 1 | Chat sessions; append-only log as the cold archive | done |
| 2 | Context Quality Monitor (CIS) + snapshot engine | done |
| 2b | Verified reconstruction + ablation benchmark | done |
| 3 | Memory recall during answering (λ-weighted fusion) | done |
| 4 | Adaptive budgeted retrieval: confidence signals, intent budgets, deeper batches | done |
| 5 | Conversation state (memory updater) + session hand-off | done |
| 6 | Long-conversation benchmark harness: baselines, ladder, ablations, scoring | done; full runs pending |
| 7 | Tests (170+), health/metrics endpoints, rate limit, chat-resume UI | done |

---

## Preliminary results

Small demonstrations on one 12-page lab report. They show the pipeline works; they are
**not** the paper's evaluation.

**Memory on vs off** (7-turn chat; the probe asks at turn 8 about a decision the user made
at turn 3, which is no longer in the recent history and is not in the PDF):

| Question | Memory ON | Memory OFF |
|---|---|---|
| Which routers did I say my report will describe? | correct (Router-0 and Router-2) | "Insufficient information found." |
| Which router will I skip in my report? | correct (Router-1) | "Insufficient information found." |
| What default gateway does PC5 use? | correct (30.30.30.0) | wrong value copied from a misread page |

**Reconstruction ablation** (5 turns, 5 probes, scored by a separate judge model): every
mode reached 0% hallucination and 100% fact recall. At only 1.5× compression the snapshot
is almost lossless, so this demo cannot separate the modes. The real comparison needs long
conversations at higher compression (Phase 6).

---

## Repository layout

```
.
├── rag-service/                 FastAPI backend (port 8003)
│   ├── main.py                  app entry point
│   ├── dev_token.py             prints a development access token
│   ├── app/
│   │   ├── api/routes/          query, upload, chats, pdf preview
│   │   ├── core/                settings, auth, startup
│   │   ├── cache/               Redis answer + embedding caches (fail-open)
│   │   ├── prompts/             generation, page vision, snapshot, reconstruction
│   │   ├── repositories/        Qdrant access
│   │   └── services/            retrieval, generation, sessions, memory (see below)
│   ├── tests/                   pytest suite (no network, models or services needed)
│   └── benchmarks/              longconv/ (long-conversation benchmark), reconstruction_eval,
│                                retrieval_eval, grounding_check + saved results
├── frontend/                    Streamlit chat UI (port 8501)
├── SNAPRAG_PLAN.md              research plan, related work, paper outline
└── a.txt                        original design notes
```

Memory-related services:

| File | Role |
|---|---|
| `session_store.py` | chats as append-only JSONL (the cold archive) |
| `context_monitor.py` | CIS signals and the snapshot trigger |
| `snapshot_engine.py` | facet extraction, anchors, hashing; runs after each turn |
| `snapshot_store.py` | snapshot files, CIS log, snapshot index |
| `reconstruction.py` | reconstruct + verification cascade; ablation modes |
| `memory_retriever.py` | λ weight, snapshot recall, fusion with document context |
| `memory_state.py` | conversation state (memory updater), hand-off document |
| `history.py` | the turns shown to the LLM (last turns + those not yet in a snapshot) |
| `abrag.py` | AB-RAG: the three confidence signals and the extra-batch loop |
| `answering.py` | generate + check one attempt (shared by batch and streaming) |

---

## Setup

Requirements:

- Python 3.10+
- Docker (for Redis and Qdrant)
- An OpenAI-compatible Chat Completions endpoint that accepts image input. The author's
  setup is a separate LLM gateway (not in this repository) in front of Google Gemini.
- Optional: [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki), used only when
  the vision model is unavailable

### 1. Install

```powershell
python -m venv venv
.\venv\Scripts\pip install -r rag-service\requirements.txt -r frontend\requirements.txt
```

On macOS/Linux use `venv/bin/` instead of `.\venv\Scripts\`.

### 2. Start Redis and Qdrant

```powershell
docker run -d --name snaprag-redis  -p 6379:6379 redis
docker run -d --name snaprag-qdrant -p 6333:6333 -v snaprag-qdrant:/qdrant/storage qdrant/qdrant
```

The volume keeps indexed documents if the Qdrant container is ever removed. Afterwards,
`docker start snaprag-redis snaprag-qdrant` is enough.

### 3. Create `.env` in the project root

```ini
# Required
JWT_SECRET_KEY=change-me-to-a-long-random-string
LLM_GATEWAY_URL=http://localhost:8006/v1
LLM_GATEWAY_TOKEN=your-gateway-or-api-token

# Local services (the defaults are Docker hostnames)
QDRANT_HOST=localhost
REDIS_HOST=localhost

# Optional
LLM_MODEL=gemini-3.5-flash-lite
LLM_REASONING_EFFORT=low
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=snaprag
```

`.env` is git-ignored. Never commit real keys.

---

## Running

Start the LLM endpoint first, then the services below, each in its own terminal.

**Backend** (port 8003):

```powershell
cd rag-service
..\venv\Scripts\uvicorn.exe main:app --port 8003 --reload
```

Run only one backend at a time. Two on the same port serve requests unpredictably.

**Frontend** (port 8501):

```powershell
cd frontend
..\venv\Scripts\streamlit.exe run app.py
```

Open http://localhost:8501 and either:

- **Start free demo**: one click creates a limited guest session (no token needed), or
- **I have an access token**: paste an unlimited token for yourself:

  ```powershell
  cd rag-service
  ..\venv\Scripts\python.exe dev_token.py
  ```

  Setting `SNAPRAG_TOKEN` in the environment signs the app in automatically.

Upload a PDF and ask questions. Each conversation is a chat: follow-ups use recent
history, and older turns are compressed into snapshots and recalled as verified memory.
The frontend finds the backend through `SNAPRAG_API_URL` (default
`http://localhost:8003`); on Streamlit Cloud, set it in the app's secrets.

### Demo limits

Guest sessions are limited by the backend (not just the UI), so a public demo can't run
up LLM costs:

| Limit | Default | Setting |
|---|---|---|
| Questions per guest | 7 | `GUEST_QUERY_LIMIT` |
| PDF uploads per guest | 2 | `GUEST_UPLOAD_LIMIT` |
| Pages per guest PDF | 40 | `GUEST_MAX_PAGES` |
| New guest sessions per visitor (IP) per day | 3 | `GUEST_SESSIONS_PER_IP` |
| Guest token lifetime | 24 h | `GUEST_TOKEN_HOURS` |

A question or upload that fails is not counted. Every user, guest or not, sees only the
PDFs they uploaded. Set `GUEST_ACCESS_ENABLED=false` to turn guest access off.

---

## API

All endpoints except `GET /` and `POST /auth/guest` need `Authorization: Bearer <token>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | health check |
| POST | `/auth/guest` | start a limited demo session, returns a guest token |
| GET | `/auth/usage` | the caller's remaining demo allowance |
| POST | `/upload` | upload and index a PDF (max 50 MB) |
| GET | `/documents` | list the caller's documents and their active one |
| POST | `/documents/active/{name}` | set the active document |
| GET | `/documents/active` | get the active document |
| GET | `/pdf/{filename}` | view a PDF the caller uploaded |
| POST | `/query` | answer a question (JSON) |
| POST | `/query/stream` | answer with server-sent events: `step`, `citations`, `token`, `verification`, `revised`, `error`, `done` (a `revised` event replaces the streamed answer) |
| POST | `/chats` | start a chat, returns `chat_id` |
| GET | `/chats` | list the caller's chats |
| GET | `/chats/{id}/messages` | the chat's recorded turns |
| GET | `/chats/{id}/memory` | the chat's CIS log, snapshots and conversation state |
| GET | `/chats/{id}/handoff` | where the conversation stands (resume document) |
| GET | `/healthz` | dependency status (503 when Qdrant, storage or the LLM config is down) |
| GET | `/metrics` | Prometheus text: queries, LLM calls, corrections, latency |

Query body: `{"question": "...", "chat_id": "<optional>"}`. Without `chat_id` nothing is
stored and no memory is used. With it, the `/query` response includes `retrieved_memory`
(λ, recalled snapshots, verified sentences and their sources).

---

## Research settings

All are environment variables (see `rag-service/app/core/config.py`):

| Setting | Default | Meaning |
|---|---|---|
| `CQM_W_GROUNDING` / `_TOKEN` / `_COHERENCE` / `_CONTRADICTION` | 0.35 / 0.25 / 0.25 / 0.15 | CIS weights |
| `CQM_TRIGGER_HEADROOM` | 0.20 | snapshot when less headroom remains |
| `CQM_TRIGGER_CIS` | 0.55 | snapshot when CIS falls below |
| `MEMORY_BUDGET_TOKENS` | 2000 | raw-conversation budget behind "headroom" |
| `SNAPSHOT_EXEMPLARS` | 3 | earlier snapshots shown to the extractor (0 = off) |
| `LAMBDA_BETA` / `LAMBDA_THETA` | 0.08 / 8 | memory-weight curve |
| `LAMBDA_FIXED` | unset | constant λ instead of the curve (ablation) |
| `MEMORY_CONTEXT_TOKENS` | 600 | memory budget at λ = 1 |
| `SMS_TOP_K` / `SMS_MIN_SIMILARITY` | 3 / 0.5 | snapshots recalled per question |
| `MEMORY_RECALL_MODE` | `archive` | `off` (no-memory baseline), `state` (conversation state only) or `snapshot`, `evidence`, `verify`, `archive` |
| `MEMORY_STATE_ENABLED` | true | show the conversation state on every turn |
| `STATE_MAX_TOKENS` / `HANDOFF_MAX_TOKENS` / `SESSION_GAP_SECONDS` | 300 / 600 / 1800 | state block size, hand-off size, idle time that makes a session "resumed" |
| `HISTORY_TURNS` / `HISTORY_MAX_TOKENS` / `HISTORY_ANSWER_CHARS` | 3 / 1500 / 800 | last turns always shown; turns after the newest snapshot are added up to the token budget; answers are shortened |
| `STOP_CONFIDENCE` / `MAX_EXTRA_BATCHES` | 0.80 / 3 | AB-RAG threshold; cap on extra batches (0 = fixed-k RAG). The per-intent budget is in `query_router.py` |
| `CONF_W_MODEL` / `CONF_W_SIMILARITY` / `CONF_W_SPREAD` | 0.50 / 0.25 / 0.25 | weights of the three confidence signals |
| `CQM_FIXED_INTERVAL` / `CQM_CHECK_CONTRADICTION` | 0 / true | ablation: snapshot every N turns; the per-turn contradiction call |
| `QUERIES_PER_MINUTE` | 20 | per-user request rate (0 = unlimited) |
| `MEMORY_ROOT` | unset | keep chats and memory elsewhere than `STORAGE_DIR` (benchmarks use a temp folder) |
| `RECONSTRUCTION_MODEL` | `LLM_MODEL` | model that rebuilds and verifies memory |
| `PDF_VISION_ENABLED` | true | vision model for image-heavy pages |

---

## Benchmark

The reconstruction ablation runs the four modes (`snapshot → evidence → verify → archive`)
and scores them with a separate judge model:

```powershell
cd rag-service
..\venv\Scripts\python.exe -m benchmarks.reconstruction_eval --demo
```

`--demo` builds a short chat in temporary storage, so real chats are never touched. To
evaluate an existing chat that already has a snapshot:

```powershell
..\venv\Scripts\python.exe -m benchmarks.reconstruction_eval --user dev-user --chat <chat_id> --question "..."
```

| Metric | Definition |
|---|---|
| Hallucination rate | returned sentences that neither the archive nor the anchored pages support, divided by returned sentences |
| Fact recall | reference facts from the archive that the memory states, divided by reference facts |
| Memory tokens | size of the memory that would enter the prompt |
| Compression | snapshot source tokens divided by snapshot tokens |

Results are saved as JSON in `rag-service/benchmarks/results/`. Expect rate-limit pauses on
free LLM tiers; the script waits and retries.

### Long-conversation benchmark (SnapRAG-LongConv)

A scripted 100-turn conversation about the torts book: document questions fill the context, the
user plants 14 facts about their own coursework (a deadline, a scope, a correction, ...), and 24
probes later ask for them at four distances (near, mid, far, very far), plus questions whose right
answer is "you have not told me" and traps built on a similar case. Scoring is by regular
expression, so it is the same on every run. Variants (`--list`): the baselines `no_memory`,
`full_history` and `summary_buffer`; the reconstruction ladder `snapshot` → `evidence` → `verify`
→ `archive`; `state_only` and `full`; and ablations of `full` (fixed λ, no exemplars, fixed
snapshot interval, fixed-k retrieval). Output: accuracy with bootstrap intervals by kind and
distance, memory loss (near minus very far), prompt tokens, LLM calls and seconds per turn,
document page-hit rate, and paired differences against `no_memory`.

```powershell
cd rag-service
..\venv\Scripts\python.exe -m benchmarks.longconv.run --dry-run                # checks the harness, no services
..\venv\Scripts\python.exe -m benchmarks.longconv.run --variants full,no_memory --turns 80
..\venv\Scripts\python.exe -m benchmarks.longconv.run --variants main --turns 100 --repeats 3 --pause 2
```

Use at least 80 turns. A run costs roughly `turns × 5` LLM calls per variant plus memory upkeep;
it saves after every variant and `--resume <file>` continues an interrupted run. LCM, Mem0, A-MEM
and LLMLingua are not included: they are separate systems and are not re-implemented here.

### Tests

```powershell
..\venv\Scripts\pip.exe install -r rag-service\requirements-dev.txt
cd rag-service
..\venv\Scripts\python.exe -m pytest
```

The suite needs no network, model or service: the LLM, embeddings and Qdrant are scripted stand-ins.

### Document-QA retrieval check

A long multi-part question is split by one planner call into short topic queries; each runs
dense + BM25 + RRF + rerank on its own, and the rankings are merged round-robin so every topic
reaches the LLM. `retrieval_eval.py` checks, for five hard questions on a 717-page torts book,
that the context holds a page from every section the question needs. It makes one call per
question, or none with `--no-llm`:

```powershell
..env\Scripts\python.exe benchmarks/retrieval_eval.py --no-llm
..env\Scripts\python.exe benchmarks/grounding_check.py res-ipsa-loquitur
```

`grounding_check.py` answers a question end to end and checks that every case name and
`(PDF p. N)` citation in the answer occurs in the context the LLM was given.

---

## Data and privacy

- Uploaded PDFs, indexes, chats, snapshots and logs live in `rag-service/storage/`
  (git-ignored).
- Questions, document passages and page images are sent to the configured LLM provider.
- With `LANGSMITH_TRACING=true`, prompts and retrieved chunks are also sent to LangSmith.
