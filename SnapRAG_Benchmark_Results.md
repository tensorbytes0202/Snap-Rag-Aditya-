# Document-QA architecture comparison

Document: **A Suitable Boy Vikram Seth.pdf** | LLM: **gemini-3.5-flash-lite** | run 2026-10-22 11:12 | 30 questions

## Architectures

| Name | What it is | Source |
|---|---|---|
| `closed_book` | no retrieval | baseline |
| `dense_k5` | dense top-5 | naive RAG |
| `bm25_k5` | BM25 top-5 | baseline |
| `hybrid_k5` | dense + BM25 (RRF) top-5 | standard hybrid RAG |
| `hybrid_rerank_k5` | hybrid + cross-encoder top-5 | standard advanced RAG |
| `hybrid_rerank_k10` | hybrid + cross-encoder top-10 | standard advanced RAG, more pages |
| `hyde` | retrieve with a hypothetical answer | HyDE, Gao et al. 2022 (arXiv 2212.10496) |
| `crag` | grade the pages, re-search if none is relevant | CRAG, Yan et al. 2024 (arXiv 2401.15884); no web search, no strip refinement |
| `self_rag` | one answer per page, a critic picks one | Self-RAG, Asai et al. 2023 (arXiv 2310.11511); prompted, not the fine-tuned model |
| `ircot` | reasoning steps interleaved with retrieval | IRCoT, Trivedi et al. 2022 (arXiv 2212.10509); 3 steps, no few-shot demos |
| `adaptive_rag` | route by question complexity | Adaptive-RAG, Jeong et al. 2024 (arXiv 2403.14403); LLM prompt instead of the trained classifier |
| `abrag` | answer, score confidence, read more while it is low | AB-RAG, arXiv 2606.29090 (known from its summary); same confidence signals, one fixed first batch and budget |
| `snaprag_fixed` | SnapRAG retrieval + verification, AB-RAG off | this work |
| `snaprag` | SnapRAG: intent-sized batches + AB-RAG | this work |

All architectures use the same index, embedder, cross-encoder, answer prompt and LLM. Questions: 3 aggregation, 6 detail, 3 false_premise, 5 inference, 10 multi_hop, 3 numeric (3 unanswerable or false-premise). Scoring is by fixed patterns, not by an LLM judge. Intervals are bootstrap 95% over questions.

## Dataset

30 questions written by reading the book; every answerable question lists the facts an answer needs and the gold pages that hold them (57 distinct pages, from page 9 to 428). Multi-hop questions need facts from two or more pages that are not neighbours.

| Question type | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| aggregation | 1 | 1 | 1 | 3 |
| detail | 1 | 5 | 0 | 6 |
| false_premise | 0 | 0 | 3 | 3 |
| inference | 0 | 2 | 3 | 5 |
| multi_hop | 0 | 4 | 6 | 10 |
| numeric | 1 | 2 | 0 | 3 |
| **all** | 3 | 14 | 13 | 30 |

## Metrics

- **Accuracy**: share of questions answered correctly. An answerable question is correct when the answer contains every required fact (fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer declines or states the correction.
- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions).
- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).
- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).
- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.
- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.
- **LLM calls, pages read, context tokens, seconds**: cost of one question.
- **Intervals** are percentile bootstrap 95% intervals over questions (2000 resamples, seed 0); comparisons between architectures are paired on the same questions, with an exact two-sided sign test on the questions where they differ.

## Run settings

| Setting | Value |
|---|---|
| `LLM_MODEL` | gemini-3.5-flash-lite |
| `LLM_REASONING_EFFORT` | low |
| `EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 |
| `CROSS_ENCODER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `PARENT_CHUNK_SIZE` | 3000 |
| `CHILD_CHUNK_SIZE` | 800 |
| `STOP_CONFIDENCE` | 0.8 |
| `MAX_EXTRA_BATCHES` | 3 |
| `CONF_W_MODEL` | 0.5 |
| `CONF_W_SIMILARITY` | 0.25 |
| `CONF_W_SPREAD` | 0.25 |
| `temperature` | 0 |

## Summary

Accuracy ranking: `ircot` 70%; `abrag` 70%; `snaprag` 70%; `hybrid_rerank_k5` 67%; `hybrid_rerank_k10` 67%; `dense_k5` 63%; `hybrid_k5` 63%; `adaptive_rag` 63%; `snaprag_fixed` 63%; `crag` 57%; `bm25_k5` 53%; `hyde` 53%; `self_rag` 43%; `closed_book` 7%.

- SnapRAG against the best published method re-implemented here (`ircot`): 70.0% against 70.0%, +0.0 pp (95% interval -17 to +17 pp; won 3, lost 3, tied 24 of 30 questions): no clear difference.
- Against the extra AB-RAG batches (SnapRAG without them): +6.7 pp (95% interval -13 to +23 pp; won 5, lost 3, tied 22 of 30 questions): no clear difference.
- Against the intent-aware first batch and budget (AB-RAG with one fixed batch for every question): +0.0 pp (95% interval -20 to +20 pp; won 4, lost 4, tied 22 of 30 questions): no clear difference.
- Against a standard hybrid + cross-encoder pipeline: +3.3 pp (95% interval -13 to +20 pp; won 4, lost 3, tied 23 of 30 questions): no clear difference.
- Cost: `snaprag` makes 3.7 LLM calls per question; `hyde` 2.0, `crag` 2.3, `self_rag` 4.0, `ircot` 3.3, `adaptive_rag` 3.0, `abrag` 3.5.

## Answer quality

Accuracy = every required fact is in the answer (unanswerable questions: the answer declines or states the correction).

| Architecture | Questions answered | Accuracy, all questions (95% CI) | vs `dense_k5` | Answerable only | Mean fact score | Unanswerable handled | Refused an answerable question |
|---|---|---|---|---|---|---|---|
| `closed_book` | 30/30 | 6.7% (0-17) | -56.7 pp | 7.4% (0-19) | 14.2% | 0.0% (0-0) | 0.0% |
| `dense_k5` | 30/30 | 63.3% (47-80) | - | 59.3% (41-78) | 79.6% | 100.0% (100-100) | 3.7% |
| `bm25_k5` | 30/30 | 53.3% (37-70) | -10.0 pp | 48.1% (30-67) | 71.0% | 100.0% (100-100) | 11.1% |
| `hybrid_k5` | 30/30 | 63.3% (47-80) | +0.0 pp | 59.3% (41-78) | 75.9% | 100.0% (100-100) | 7.4% |
| `hybrid_rerank_k5` | 30/30 | 66.7% (50-83) | +3.3 pp | 63.0% (44-81) | 79.0% | 100.0% (100-100) | 11.1% |
| `hybrid_rerank_k10` | 30/30 | 66.7% (50-83) | +3.3 pp | 63.0% (44-81) | 82.1% | 100.0% (100-100) | 3.7% |
| `hyde` | 30/30 | 53.3% (37-73) | -10.0 pp | 48.1% (30-67) | 66.0% | 100.0% (100-100) | 14.8% |
| `crag` | 30/30 | 56.7% (40-73) | -6.7 pp | 51.9% (33-70) | 74.1% | 100.0% (100-100) | 7.4% |
| `self_rag` | 30/30 | 43.3% (27-60) | -20.0 pp | 37.0% (19-56) | 62.3% | 100.0% (100-100) | 22.2% |
| `ircot` | 30/30 | **70.0% (53-87)** | +6.7 pp | 66.7% (48-85) | 85.2% | 100.0% (100-100) | 3.7% |
| `adaptive_rag` | 30/30 | 63.3% (47-80) | +0.0 pp | 59.3% (41-78) | 77.8% | 100.0% (100-100) | 11.1% |
| `abrag` | 30/30 | **70.0% (53-87)** | +6.7 pp | 66.7% (48-85) | 82.7% | 100.0% (100-100) | 7.4% |
| `snaprag_fixed` | 30/30 | 63.3% (47-80) | +0.0 pp | 59.3% (41-78) | 77.2% | 100.0% (100-100) | 7.4% |
| `snaprag` | 30/30 | **70.0% (53-87)** | +6.7 pp | 66.7% (48-85) | 83.3% | 100.0% (100-100) | 7.4% |

## Retrieval quality

Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. These are measured on answerable questions only.

| Architecture | Context recall (95% CI) | Page hit | Gold pages read | Pages read (mean) | Context (characters) |
|---|---|---|---|---|---|
| `dense_k5` | 80.9% (70-91) | 88.9% (78-100) | 67.3% | 5.0 | 6,632 |
| `bm25_k5` | 72.2% (58-85) | 88.9% (78-100) | 64.2% | 5.0 | 6,623 |
| `hybrid_k5` | 78.4% (66-90) | 88.9% (78-100) | 72.2% | 5.0 | 6,667 |
| `hybrid_rerank_k5` | 81.5% (69-93) | 88.9% (74-100) | 72.8% | 5.0 | 6,690 |
| `hybrid_rerank_k10` | 85.8% (74-95) | 92.6% (81-100) | 80.2% | 10.0 | 12,340 |
| `hyde` | 67.9% (52-82) | 85.2% (70-96) | 61.7% | 5.0 | 6,833 |
| `crag` | 77.8% (65-90) | 92.6% (81-100) | 60.5% | 2.4 | 3,547 |
| `self_rag` | 74.1% (59-87) | 85.2% (70-96) | 67.3% | 3.0 | 4,475 |
| `ircot` | **88.9% (79-97)** | 96.3% (89-100) | 77.8% | 4.3 | 6,374 |
| `adaptive_rag` | 81.5% (69-93) | 88.9% (74-100) | 74.1% | 4.5 | 6,207 |
| `abrag` | 85.2% (73-95) | 92.6% (81-100) | 76.5% | 7.0 | 10,002 |
| `snaprag_fixed` | 79.0% (65-90) | 92.6% (81-100) | 72.2% | 5.4 | 8,016 |
| `snaprag` | 85.8% (74-95) | 96.3% (89-100) | 77.2% | 6.2 | 9,153 |

### Context recall by question type

| Architecture | detail (n=6) | numeric (n=3) | aggregation (n=3) | multi_hop (n=10) | inference (n=5) |
|---|---|---|---|---|---|
| `dense_k5` | 100.0% | 50.0% | 88.9% | 73.3% | 86.7% |
| `bm25_k5` | 55.6% | 50.0% | 66.7% | 80.0% | 93.3% |
| `hybrid_k5` | 83.3% | 50.0% | 100.0% | 70.0% | 93.3% |
| `hybrid_rerank_k5` | 100.0% | 50.0% | 88.9% | 71.7% | 93.3% |
| `hybrid_rerank_k10` | 100.0% | 66.7% | 88.9% | 78.3% | 93.3% |
| `hyde` | 72.2% | 33.3% | 100.0% | 60.0% | 80.0% |
| `crag` | 100.0% | 50.0% | 77.8% | 71.7% | 80.0% |
| `self_rag` | 83.3% | 50.0% | 88.9% | 61.7% | 93.3% |
| `ircot` | 100.0% | 50.0% | 100.0% | 88.3% | 93.3% |
| `adaptive_rag` | 100.0% | 50.0% | 88.9% | 71.7% | 93.3% |
| `abrag` | 100.0% | 66.7% | 88.9% | 76.7% | 93.3% |
| `snaprag_fixed` | 83.3% | 33.3% | 88.9% | 80.0% | 93.3% |
| `snaprag` | 83.3% | 66.7% | 100.0% | 85.0% | 93.3% |

## Cost and speed

Context tokens are the characters of the pages the LLM read divided by 4 (an estimate; the answer prompt and the extra calls of the longer pipelines add more). Seconds include the wait for the request-rate limit.

| Architecture | LLM calls per question | Accuracy per LLM call | Pages read (mean) | Context tokens (estimate) | Median seconds | 95th percentile seconds | Errors |
|---|---|---|---|---|---|---|---|
| `closed_book` | 1.0 | 6.7 pp | 0.0 | 0 | 4.2 | 144.1 | 0 |
| `dense_k5` | 1.0 | 63.3 pp | 5.0 | 1,658 | 3.5 | 51.8 | 0 |
| `bm25_k5` | 1.0 | 53.3 pp | 5.0 | 1,656 | 9.7 | 42.2 | 0 |
| `hybrid_k5` | 1.0 | 63.3 pp | 5.0 | 1,667 | 12.0 | 25.1 | 0 |
| `hybrid_rerank_k5` | 1.0 | 66.7 pp | 5.0 | 1,672 | 11.3 | 20.0 | 0 |
| `hybrid_rerank_k10` | 1.0 | 66.7 pp | 10.0 | 3,085 | 12.5 | 42.6 | 0 |
| `hyde` | 2.0 | 26.7 pp | 5.0 | 1,708 | 33.6 | 71.6 | 0 |
| `crag` | 2.3 | 25.0 pp | 2.4 | 887 | 52.4 | 61.9 | 0 |
| `self_rag` | 4.0 | 10.8 pp | 3.0 | 1,119 | 67.2 | 233.4 | 0 |
| `ircot` | 3.3 | 21.4 pp | 4.3 | 1,594 | 53.1 | 110.0 | 0 |
| `adaptive_rag` | 3.0 | 21.1 pp | 4.5 | 1,552 | 56.0 | 117.0 | 0 |
| `abrag` | 3.5 | 20.2 pp | 7.0 | 2,501 | 67.2 | 284.7 | 0 |
| `snaprag_fixed` | 3.1 | 20.7 pp | 5.4 | 2,004 | 35.1 | 57.9 | 0 |
| `snaprag` | 3.7 | 18.9 pp | 6.2 | 2,288 | 59.5 | 107.8 | 0 |

## Self-verification (architectures with a confidence check)

Verified = the checker found the answer supported by the retrieved text. A useful check separates answers that are right from answers that are wrong: accuracy should be higher among the verified ones.

| Architecture | Mean confidence | Answers verified | Accuracy when verified | Accuracy when not verified |
|---|---|---|---|---|
| `abrag` | 0.86 | 100.0% | 70.0% (53-87) | - |
| `snaprag_fixed` | 0.86 | 96.7% | 65.5% (48-83) | 0.0% (0-0) |
| `snaprag` | 0.88 | 100.0% | 70.0% (53-87) | - |

`adaptive_rag` routed 0 questions to no retrieval, 17 to single-step retrieval and 13 to multi-step retrieval.

## Accuracy by question type

| Architecture | detail (n=6) | numeric (n=3) | aggregation (n=3) | multi_hop (n=10) | inference (n=5) | false_premise (n=3) |
|---|---|---|---|---|---|---|
| `closed_book` | 0.0% | 33.3% | 0.0% | 0.0% | 20.0% | 0.0% |
| `dense_k5` | 100.0% | 33.3% | 66.7% | 40.0% | 60.0% | 100.0% |
| `bm25_k5` | 50.0% | 33.3% | 66.7% | 30.0% | 80.0% | 100.0% |
| `hybrid_k5` | 83.3% | 33.3% | 100.0% | 30.0% | 80.0% | 100.0% |
| `hybrid_rerank_k5` | 100.0% | 33.3% | 66.7% | 40.0% | 80.0% | 100.0% |
| `hybrid_rerank_k10` | 100.0% | 66.7% | 66.7% | 30.0% | 80.0% | 100.0% |
| `hyde` | 66.7% | 33.3% | 100.0% | 30.0% | 40.0% | 100.0% |
| `crag` | 100.0% | 33.3% | 66.7% | 30.0% | 40.0% | 100.0% |
| `self_rag` | 50.0% | 33.3% | 66.7% | 10.0% | 60.0% | 100.0% |
| `ircot` | 100.0% | 33.3% | 100.0% | 40.0% | 80.0% | 100.0% |
| `adaptive_rag` | 100.0% | 33.3% | 66.7% | 30.0% | 80.0% | 100.0% |
| `abrag` | 100.0% | 66.7% | 66.7% | 40.0% | 80.0% | 100.0% |
| `snaprag_fixed` | 83.3% | 33.3% | 66.7% | 50.0% | 60.0% | 100.0% |
| `snaprag` | 83.3% | 66.7% | 100.0% | 50.0% | 60.0% | 100.0% |

## Accuracy by difficulty

| Architecture | easy (n=3) | medium (n=14) | hard (n=13) |
|---|---|---|---|
| `closed_book` | 33.3% | 7.1% | 0.0% |
| `dense_k5` | 100.0% | 64.3% | 53.8% |
| `bm25_k5` | 100.0% | 35.7% | 61.5% |
| `hybrid_k5` | 100.0% | 57.1% | 61.5% |
| `hybrid_rerank_k5` | 100.0% | 64.3% | 61.5% |
| `hybrid_rerank_k10` | 100.0% | 71.4% | 53.8% |
| `hyde` | 100.0% | 57.1% | 38.5% |
| `crag` | 100.0% | 57.1% | 46.2% |
| `self_rag` | 100.0% | 35.7% | 38.5% |
| `ircot` | 100.0% | 71.4% | 61.5% |
| `adaptive_rag` | 100.0% | 64.3% | 53.8% |
| `abrag` | 100.0% | 71.4% | 61.5% |
| `snaprag_fixed` | 66.7% | 64.3% | 61.5% |
| `snaprag` | 100.0% | 78.6% | 53.8% |

## `snaprag` against every other architecture (paired, same questions)

Each cell: mean difference in percentage points (95% bootstrap interval); questions won / lost / tied; 'better' or 'worse' only when the interval excludes zero. The sign test is an exact two-sided binomial test on the questions where the two architectures differ.

| Compared with | Accuracy difference | Fact-score difference | Context-recall difference |
|---|---|---|---|
| `closed_book` | +63.3 pp (+47 to +80); 19/0/11; sign test p=0.000; better | +72.2 pp (+58 to +86); 26/0/4; better | +85.8 pp (+74 to +95); 25/0/2; better |
| `dense_k5` | +6.7 pp (-10 to +27); 5/3/22; sign test p=0.727; no clear difference | +3.3 pp (-8 to +15); 5/3/22; no clear difference | +4.9 pp (-9 to +19); 6/3/18; no clear difference |
| `bm25_k5` | +16.7 pp (+0 to +37); 7/2/21; sign test p=0.180; no clear difference | +11.1 pp (+1 to +23); 7/2/21; better | +13.6 pp (+2 to +27); 6/1/20; better |
| `hybrid_k5` | +6.7 pp (-10 to +23); 4/2/24; sign test p=0.688; no clear difference | +6.7 pp (-2 to +17); 5/2/23; no clear difference | +7.4 pp (-2 to +19); 4/1/22; no clear difference |
| `hybrid_rerank_k5` | +3.3 pp (-13 to +20); 4/3/23; sign test p=1.000; no clear difference | +3.9 pp (-8 to +16); 5/3/22; no clear difference | +4.3 pp (-9 to +17); 5/3/19; no clear difference |
| `hybrid_rerank_k10` | +3.3 pp (-13 to +20); 4/3/23; sign test p=1.000; no clear difference | +1.1 pp (-11 to +12); 4/3/23; no clear difference | +0.0 pp (-12 to +12); 3/3/21; no clear difference |
| `hyde` | +16.7 pp (+0 to +33); 6/1/23; sign test p=0.125; no clear difference | +15.6 pp (+5 to +26); 9/1/20; better | +17.9 pp (+7 to +30); 8/1/18; better |
| `crag` | +13.3 pp (-7 to +33); 7/3/20; sign test p=0.344; no clear difference | +8.3 pp (-4 to +21); 8/3/19; no clear difference | +8.0 pp (-6 to +20); 8/3/16; no clear difference |
| `self_rag` | +26.7 pp (+10 to +47); 9/1/20; sign test p=0.021; better | +18.9 pp (+8 to +32); 10/1/19; better | +11.7 pp (+0 to +25); 6/2/19; no clear difference |
| `ircot` | +0.0 pp (-17 to +17); 3/3/24; sign test p=1.000; no clear difference | -1.7 pp (-12 to +7); 3/3/24; no clear difference | -3.1 pp (-14 to +6); 2/3/22; no clear difference |
| `adaptive_rag` | +6.7 pp (-13 to +23); 5/3/22; sign test p=0.727; no clear difference | +5.0 pp (-7 to +17); 6/3/21; no clear difference | +4.3 pp (-9 to +17); 5/3/19; no clear difference |
| `abrag` | +0.0 pp (-20 to +20); 4/4/22; sign test p=1.000; no clear difference | +0.6 pp (-11 to +11); 5/4/21; no clear difference | +0.6 pp (-12 to +12); 4/3/20; no clear difference |
| `snaprag_fixed` | +6.7 pp (-13 to +23); 5/3/22; sign test p=0.727; no clear difference | +5.6 pp (-4 to +16); 5/3/22; no clear difference | +6.8 pp (-2 to +18); 4/1/22; no clear difference |

## Questions `snaprag` got wrong (9)

- **q06** (multi_hop, medium): Where does Arun Mehra work, what is his brother Varun studying, and where do the two of them live?  
  expected: Arun is an executive at Bentsen & Pryce in Calcutta; Varun studies mathematics at Calcutta University and lives with Arun and Meenakshi in Calcutta.  
  answered: - Arun Mehra's workplace: Not found in the retrieved text: the specific place where Arun Mehra works (though he goes to an office, PDF p. 381). - Varun is studying mathematics at Calcutta University (PDF p. 22). - They live in Arun and Meenakshi's small ground
- **q08** (inference, hard): What does Arun look down on about the Kapoors and the wedding celebrations?  
  expected: He finds them provincial and crass: the rustic relatives and state politicians, the coloured lights in the hedge, the pomegranate juice instead of alcohol, and Maan running a cloth shop in Banaras.  
  answered: The text states: - Arun considers the Kapoors to be merely "all very well for Brahmpur society" and compares them unfavourably to his in-laws, noting that Pran's brother ran "a cloth shop in Banaras" (PDF p. 26). - He looks down on the wedding celebrations as 
- **q13** (detail, medium): Who saved the convoy Kedarnath was travelling in when it was attacked near the Ravi Bridge?  
  expected: A Gurkha regiment coming from the other side opened fire and the attackers fled.  
  answered: Insufficient information found.
- **q33** (multi_hop, hard): What orders did Home Minister L.N. Agarwal give District Magistrate Krishan Dayal about Misri Mandi, and what practical objection did the DM raise?  
  expected: He told him to guard the entrances of the Govind Shoe Mart and disperse pickets, reading Section 144 and firing if necessary; the DM objected that police were tied up at the Shiva Temple site.  
  answered: **The text states:** Home Minister L.N. Agarwal ordered District Magistrate Krishan Dayal to guard the entrance of the Govind Shoe Mart, disperse any pickets forthwith, read out Section 144 of the Criminal Procedure Code if necessary, and then fire if the crow
- **q36** (multi_hop, hard): How was the police action at Baitar House stopped, and who carried the appeal to the Chief Minister?  
  expected: Zainab wrote a letter that Murtaza Ali and little Hassan carried to the Chief Minister, who phoned Agarwal and ordered him to call it off and make the police leave.  
  answered: **The text states:** * Murtaza Ali carried the appeal to the Chief Minister via a letter ([Source: A Suitable Boy Vikram Seth.pdf, PDF page 241]). * The Chief Minister stopped the action by calling the Home Minister (Agarwal) and ordering him to "call it off a
- **q41** (numeric, medium): How many people did Begum Abida Khan say were being dispossessed by the bill?  
  expected: Eight lakh people.  
  answered: Not found in the retrieved text: the number of people being dispossessed by the bill (though the text notes that hundreds of people depend on the establishments of larger landlords, PDF p. 244).
- **q46** (inference, hard): Why was Ustad Majeed Khan upset about the Shiva Temple, and what did he ask Veena to tell her father?  
  expected: The temple being built in Chowk near his house would let the conch compete with the muezzin's call to prayer; he asked her to tell her father, the Minister, to help protect their religion.  
  answered: Based on the provided context: * **Why Ustad Majeed Khan was upset about the Shiva Temple:** The text states he was concerned because once bhajans start being sung from the temple while they are at prayer, "it can only end badly" (PDF p. 266). * **What he aske
- **q48** (multi_hop, hard): Why is Ishaq Khan trying to get his brother-in-law moved, and how do artists' grades matter?  
  expected: He wants his sister's husband, a sarangi player and staff artist, moved from AIR Lucknow to AIR Brahmpur; the Station Director would not swap a B-plus artist like Rafiq for a B-grade one.  
  answered: ### Reasons for the Brother-in-Law's Move * **Support for Younger Brothers:** After their father's death, Ishaq Khan and his sister had to support their three young brothers (PDF p. 272, 356). * **Need for Assistance:** Because of this shared family responsibi
- **q58** (multi_hop, hard): What book did Dr Ila Chattopadhyay write, and what did she think of the ghazal singer's performance she attended in Brahmpur?  
  expected: She wrote 'Metaphysical Causality', a book about Donne; she found the ghazal singer (Saeeda something) dreadful and the self-pitying verse insipid.  
  answered: The text states that Dr. Ila Chattopadhyay discussed a book about Donne called *Metaphysical Causality* (PDF p. 426), referring to it as "one of my old books" (PDF p. 425) and calling it "a very stupid book" (PDF p. 426). Not found in the retrieved text: what 

## Notes

- One document, one LLM and 30 questions written by reading the book: the intervals show how little this many questions can separate close systems. Treat differences smaller than the interval as noise.
- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.
- The book is a scanned novel; some pages have broken text, which lowers every architecture equally.
- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.
- SnapRAG's AB-RAG weights and threshold are uncalibrated defaults.


---

# Document-QA architecture comparison

Document: **A Suitable Boy Vikram Seth.pdf** | LLM: **none (retrieval only)** | run 2026-10-22 10:45 | 75 questions

## Architectures

| Name | What it is | Source |
|---|---|---|
| `dense_k5` | dense top-5 | naive RAG |
| `bm25_k5` | BM25 top-5 | baseline |
| `hybrid_k5` | dense + BM25 (RRF) top-5 | standard hybrid RAG |
| `hybrid_rerank_k5` | hybrid + cross-encoder top-5 | standard advanced RAG |
| `hybrid_rerank_k10` | hybrid + cross-encoder top-10 | standard advanced RAG, more pages |
| `snaprag_fixed` | SnapRAG retrieval + verification, AB-RAG off | this work |

All architectures use the same index, embedder and cross-encoder; no LLM answers are generated, and SnapRAG's planner and HyDE run their offline fallbacks. Questions: 6 aggregation, 16 detail, 7 false_premise, 14 inference, 23 multi_hop, 7 numeric, 2 unanswerable (9 unanswerable or false-premise). Scoring is by fixed patterns, not by an LLM judge. Intervals are bootstrap 95% over questions.

## Dataset

75 questions written by reading the book; every answerable question lists the facts an answer needs and the gold pages that hold them (114 distinct pages, from page 9 to 433). Multi-hop questions need facts from two or more pages that are not neighbours.

| Question type | Easy | Medium | Hard | Total |
|---|---|---|---|---|
| aggregation | 1 | 3 | 2 | 6 |
| detail | 1 | 14 | 1 | 16 |
| false_premise | 0 | 0 | 7 | 7 |
| inference | 0 | 5 | 9 | 14 |
| multi_hop | 0 | 10 | 13 | 23 |
| numeric | 2 | 4 | 1 | 7 |
| unanswerable | 0 | 0 | 2 | 2 |
| **all** | 4 | 36 | 35 | 75 |

## Metrics

- **Accuracy**: share of questions answered correctly. An answerable question is correct when the answer contains every required fact (fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer declines or states the correction.
- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions).
- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).
- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).
- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.
- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.
- **LLM calls, pages read, context tokens, seconds**: cost of one question.
- **Intervals** are percentile bootstrap 95% intervals over questions (2000 resamples, seed 0); comparisons between architectures are paired on the same questions, with an exact two-sided sign test on the questions where they differ.

## Run settings

| Setting | Value |
|---|---|
| `LLM_MODEL` | gemini-3.5-flash-lite |
| `LLM_REASONING_EFFORT` | low |
| `EMBEDDING_MODEL` | BAAI/bge-small-en-v1.5 |
| `CROSS_ENCODER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| `PARENT_CHUNK_SIZE` | 3000 |
| `CHILD_CHUNK_SIZE` | 800 |
| `STOP_CONFIDENCE` | 0.8 |
| `MAX_EXTRA_BATCHES` | 3 |
| `CONF_W_MODEL` | 0.5 |
| `CONF_W_SIMILARITY` | 0.25 |
| `CONF_W_SPREAD` | 0.25 |
| `temperature` | 0 |

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

Each cell: mean difference in percentage points (95% bootstrap interval); questions won / lost / tied; 'better' or 'worse' only when the interval excludes zero. The sign test is an exact two-sided binomial test on the questions where the two architectures differ.

| Compared with | Accuracy difference | Fact-score difference | Context-recall difference |
|---|---|---|---|
| `dense_k5` | - | - | +9.7 pp (+4 to +16); 13/1/52; better |
| `bm25_k5` | - | - | +10.5 pp (+3 to +19); 12/3/51; better |
| `hybrid_k5` | - | - | +6.9 pp (+1 to +13); 10/2/54; better |
| `hybrid_rerank_k5` | - | - | +3.5 pp (+1 to +7); 4/0/62; better |
| `hybrid_rerank_k10` | - | - | -0.5 pp (-4 to +3); 1/2/63; no clear difference |

## Notes

- One document, one LLM and 75 questions written by reading the book: the intervals show how little this many questions can separate close systems. Treat differences smaller than the interval as noise.
- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.
- The book is a scanned novel; some pages have broken text, which lowers every architecture equally.
- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.
- SnapRAG's AB-RAG weights and threshold are uncalibrated defaults.
