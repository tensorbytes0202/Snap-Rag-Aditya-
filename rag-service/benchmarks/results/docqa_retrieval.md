# Document-QA architecture comparison

Document: **A Suitable Boy Vikram Seth.pdf** | LLM: **none (retrieval only)** | run 2026-10-22 10:45 | 75 questions

## Architectures

| Name | What it is |
|---|---|
| `dense_k5` | dense top-5 |
| `bm25_k5` | BM25 top-5 |
| `hybrid_k5` | dense + BM25 (RRF) top-5 |
| `hybrid_rerank_k5` | hybrid + cross-encoder top-5 |
| `hybrid_rerank_k10` | hybrid + cross-encoder top-10 |
| `snaprag_fixed` | SnapRAG retrieval + verification, AB-RAG off |

All architectures use the same index, embedder and cross-encoder; no LLM answers are generated, and SnapRAG's planner and HyDE run their offline fallbacks. Questions: 6 aggregation, 16 detail, 7 false_premise, 14 inference, 23 multi_hop, 7 numeric, 2 unanswerable (9 unanswerable or false-premise). Scoring is by fixed patterns, not by an LLM judge. Intervals are bootstrap 95% over questions.

## Retrieval quality

Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. These are measured on answerable questions only.

| Architecture | Context recall (95% CI) | Page hit | Gold pages read | Pages read (mean) | Context (characters) |
|---|---|---|---|---|---|
| `dense_k5` | 81.2% (74-88) | 86.4% (77-94) | 69.1% | 5.0 | 6,432 |
| `bm25_k5` | 80.4% (72-88) | 89.4% (82-95) | 68.8% | 5.0 | 6,429 |
| `hybrid_k5` | 84.0% (77-91) | 90.9% (83-97) | 73.4% | 5.0 | 6,314 |
| `hybrid_rerank_k5` | 87.4% (81-93) | 95.5% (89-100) | 77.8% | 5.0 | 6,533 |
| `hybrid_rerank_k10` | **91.4% (86-96)** | 97.0% (92-100) | 85.2% | 10.0 | 12,056 |
| `snaprag_fixed` | 90.9% (85-96) | 95.5% (89-100) | 81.6% | 5.0 | 7,134 |

### Context recall by question type

| Architecture | detail (n=16) | numeric (n=7) | aggregation (n=6) | multi_hop (n=23) | inference (n=14) |
|---|---|---|---|---|---|
| `dense_k5` | 90.6% | 78.6% | 84.7% | 71.0% | 86.9% |
| `bm25_k5` | 83.3% | 54.8% | 68.1% | 83.3% | 90.5% |
| `hybrid_k5` | 87.5% | 69.0% | 90.3% | 79.0% | 92.9% |
| `hybrid_rerank_k5` | 100.0% | 69.0% | 88.9% | 79.0% | 95.2% |
| `hybrid_rerank_k10` | 100.0% | 85.7% | 94.4% | 84.1% | 95.2% |
| `snaprag_fixed` | 100.0% | 85.7% | 88.9% | 84.1% | 95.2% |

## `snaprag_fixed` against every other architecture (paired, same questions)

Each cell: mean difference in percentage points (95% interval); questions won / lost / tied; 'better' or 'worse' only when the interval excludes zero.

| Compared with | Accuracy difference | Context-recall difference |
|---|---|---|
| `dense_k5` | - | +9.7 pp (+4 to +16); 13/1/52; better |
| `bm25_k5` | - | +10.5 pp (+3 to +19); 12/3/51; better |
| `hybrid_k5` | - | +6.9 pp (+1 to +13); 10/2/54; better |
| `hybrid_rerank_k5` | - | +3.5 pp (+1 to +7); 4/0/62; better |
| `hybrid_rerank_k10` | - | -0.5 pp (-4 to +3); 1/2/63; no clear difference |

## Notes

- One document, one LLM and 75 questions written by reading the book: the intervals show how little 75 questions can separate close systems. Treat differences smaller than the interval as noise.
- The book is a scanned novel; some pages have broken text, which lowers every architecture equally.
- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.
- SnapRAG's AB-RAG weights and threshold are uncalibrated defaults.
