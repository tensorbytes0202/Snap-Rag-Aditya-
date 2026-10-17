# SnapRAG: research plan

Working title: **SnapRAG: Verified Reconstruction of Snapshot Memory for Long-Conversation Document QA**

The system is built (see `README.md`). This file keeps what the code cannot say: the claims, what
exists already, and what still has to be shown.

---

## 1. Idea

When a long conversation about documents outgrows the context, the old turns are kept twice: the
**full text in an immutable cold archive** and a **small lossy snapshot** (facts, numbers, what the user
said, entities, decisions, open questions, with `doc/page` anchors). When the old information is needed
again, text is **reconstructed** from the snapshot and **every sentence is verified** against the snapshot,
the document pages it is anchored to, and the archive. Only verified sentences enter the prompt.

*compress, reconstruct later, check against evidence, only then use.*

Around it: a quality score (CIS) decides when to snapshot; a turn-aware weight λ(t) decides how much
memory fits; the user's own statements are kept as a verified conversation state; and retrieval depth
adapts to the question (AB-RAG).

## 2. Claims and prior art

| Part | Prior art | Claim |
|---|---|---|
| Hybrid retrieval, RRF, rerank, HyDE | standard | none |
| Immutable raw archive + expand from summary | LCM | none |
| Versioned snapshots, hash chain | ChronoMem, Cognis, OSS tools | none |
| Keyword/entity lossy compression | MemCoRe, LLMLingua | none |
| Confidence-gated retry, retrieval-depth control | CRAG, Self-RAG, **AB-RAG** | none |
| **C1. Verified reconstruction** against snapshot, pages and archive | LCM, MemCoRe do not describe it | main |
| **C2. CIS trigger** (grounding, coherence, contradiction, headroom) and **turn-aware λ** | not found | small |
| **C3. Long document-QA conversation benchmark** (100 turns, page citations) | MTRAG ~8 turns; LoCoMo is personal chat | benchmark gap |
| **C4. Intent- and memory-aware AB-RAG**: intent sets first batch and budget; verified memory is evidence; conversation questions skip the documents | AB-RAG itself, not these additions | small, unvalidated |

AB-RAG: *Adaptive Budgeted Retrieval-Augmented Generation*, https://arxiv.org/abs/2606.29090, reference
code https://github.com/Killer1GhosT/adaptive-budgeted-rag (known here only from a summary; read the
paper for its exact equations before citing). Our S1 is a judged certainty, not token probabilities
(the gateway gives none), so it is a proxy and must be described as one.

Honest position: moderate novelty. The table rests on paper summaries and a limited search. Before
claiming anything, search Google Scholar, Semantic Scholar and arXiv ("verified reconstruction",
"memory hallucination", "context quality score", "intent-aware adaptive retrieval") and have a second
person review the related-work table.

## 3. Evaluation (what the benchmark has to show)

`python -m benchmarks.longconv.run`, see `README.md`.

| Claim | Evidence |
|---|---|
| C1 | ladder `snapshot` → `evidence` → `verify` → `archive`: hallucination and fact recall (`reconstruction_eval`), accuracy by distance (`longconv`) |
| memory helps | `full` vs `no_memory`, `full_history`, `summary_buffer`: paired accuracy difference, memory loss (near minus very far), tokens per turn |
| C2 | `full` vs `full_fixed_interval` (snapshot every 5 turns) and `full_fixed_lambda` |
| C4 | `full` vs `full_fixed_k` (no extra batches): accuracy, document page-hit rate, passages read, LLM calls |
| exemplars | `full` vs `full_no_exemplars`; claim it only if it helps |

Claim honestly: near full-context accuracy at fewer tokens, with fewer hallucinations from compressed
memory. Do **not** claim "most accurate". LCM, Mem0, A-MEM and LLMLingua are not implemented here;
compare against them only with their own code. The saved reconstruction demo (5 turns, 1.5x
compression) cannot separate the ladder: the real comparison needs long conversations at 4-10x
compression.

Open points: calibrate the AB-RAG weights and threshold on a held-out set; validate the LLM judge;
add a second document and a second LLM backbone; repeat runs (`--repeats 3`) for intervals.

## 4. Reading list (read in full before citing)

| Role | Paper |
|---|---|
| Closest competitor | LCM: Lossless Context Management, https://arxiv.org/html/2605.04050 |
| Retrieval depth | AB-RAG, https://arxiv.org/abs/2606.29090 |
| Motivation | Information Preservation in Prompt Compression, https://arxiv.org/html/2503.19114v2 |
| Benchmarks | LoCoMo https://arxiv.org/html/2402.17753v1, MTRAG https://arxiv.org/abs/2501.03468 |
| Baselines | Mem0 https://arxiv.org/html/2504.19413, A-MEM https://arxiv.org/html/2502.12110, LLMLingua https://arxiv.org/html/2310.05736v2 |
| Related | MemCoRe 2602.07885, ChronoMem 2607.27773, Cognis 2604.19771, CRAG 2401.15884, Self-RAG, ICAE 2307.06945, Recursive Summarization 2308.15022 |

## 5. Paper outline

Abstract, Introduction (a fact from turn 5 lost or distorted by turn 80), Related Work (table: paper,
what it does, difference), Problem setup, Method (CIS trigger, snapshot + archive, recall with λ,
**verified reconstruction**, conversation state, AB-RAG), Experiments, Results (ablation ladder is the
main figure), Limitations (LLM cost, one backbone, one document, uncalibrated proxies), Conclusion.
Write experiments first, abstract last; leave its numbers blank until results exist.
