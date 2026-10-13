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
  ├─ Document retrieval   query understanding → dense (Qdrant) + BM25 → RRF → cross-encoder rerank
  │                       → full parent pages of the top hits + a one-line-per-page document overview
  ├─ Memory recall        (inside a chat) find similar snapshots → reconstruct → verify → keep verified
  │                       sentences, within a budget of λ(t) × 600 tokens
  ├─ Answer               generated from documents + verified memory (+ last 3 turns for follow-ups)
  │                       → answer verification → confidence → follow-up questions
  └─ Record               turn appended to the chat log (the cold archive)

After every turn (background task)
  ├─ Context Quality Monitor
  │     CIS = 0.35·grounding + 0.25·headroom + 0.25·coherence − 0.15·contradiction
  │     take a snapshot if headroom < 0.20 or CIS < 0.55
  └─ Snapshot engine
        LLM extracts facts / entities / decisions / open questions from the un-snapshotted turns
        → page anchors from the turns' citations → state_hash (sha256), chained via prev_hash
        → written once to disk + indexed in Qdrant; recent snapshots guide the next extraction
```

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
| **C3. Long document-QA conversation benchmark** (100+ turns, page citations) | harness started; dataset planned |

Hybrid retrieval, hash chains, immutability and confidence gating are supporting design,
not claimed contributions (see LCM, ChronoMem, Cognis, CRAG, Self-RAG). The full plan,
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
| 4 | Confidence-gated self-correction (retry instead of discarding) | planned |
| 5 | Session hand-off + memory updater | planned |
| 6 | Long-conversation benchmark, baselines, full ablations | planned |
| 7 | Tests, polish, chat-session UI | planned |

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
│   └── benchmarks/              reconstruction_eval.py + saved results
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

**Access token** (the API has no login; this signs a 24-hour token with your
`JWT_SECRET_KEY`):

```powershell
cd rag-service
..\venv\Scripts\python.exe dev_token.py
```

**Frontend** (port 8501):

```powershell
cd frontend
..\venv\Scripts\streamlit.exe run app.py
```

Open http://localhost:8501, paste the token under *Connection*, upload a PDF, and ask
questions. Each conversation is a chat: follow-ups use recent history, and older turns are
compressed into snapshots and recalled as verified memory.

---

## API

All endpoints except `GET /` need `Authorization: Bearer <token>`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | health check |
| POST | `/upload` | upload and index a PDF (max 50 MB) |
| GET | `/documents` | list documents and the caller's active one |
| POST | `/documents/active/{name}` | set the active document |
| GET | `/documents/active` | get the active document |
| GET | `/pdf/{filename}` | view a stored PDF |
| POST | `/query` | answer a question (JSON) |
| POST | `/query/stream` | answer with server-sent events: `step`, `citations`, `token`, `error`, `done` |
| POST | `/chats` | start a chat, returns `chat_id` |
| GET | `/chats` | list the caller's chats |
| GET | `/chats/{id}/messages` | the chat's recorded turns |
| GET | `/chats/{id}/memory` | the chat's CIS log and snapshots |

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
| `MEMORY_RECALL_MODE` | `archive` | `off` (no-memory baseline) or `snapshot`, `evidence`, `verify`, `archive` |
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

---

## Data and privacy

- Uploaded PDFs, indexes, chats, snapshots and logs live in `rag-service/storage/`
  (git-ignored).
- Questions, document passages and page images are sent to the configured LLM provider.
- With `LANGSMITH_TRACING=true`, prompts and retrieved chunks are also sent to LangSmith.
