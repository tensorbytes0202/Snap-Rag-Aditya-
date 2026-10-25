# SnapRAG: status of the plan (updated 2026-10-23, 14:50)

## What stopped the experiments

The runs were made on a **free Gemini key. Its hard limit is 500 requests per day per model** (the error says
`GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue 500`; the next window opens in about 22 hours), on top of 15 requests a minute.
The plan needs about **10,000 LLM calls** for one repeat (long-conversation variants about 5.5 calls a turn x 80 turns, the 14 architectures
about 31 calls a question x 75, calibration and diagnostics on top) and about 25,000 for three repeats. At 500 a day per model that is many days,
and every retry that is rejected still burns time. Two earlier attempts also lost a day to a retry storm (the SDK retried rejected requests
outside the rate limiter); that is fixed (retries now pass the limiter, a per-day quota is not retried, a conversation stops after 5 failed turns).

Measured so far: a one-call architecture uses about 1,620 prompt and 64 completion tokens per question. With the figures below for tokens
(about 30M prompt and 1.5M completion tokens for one repeat of everything; about 70M and 3M for three repeats), a **paid key would finish the plan in a few
hours for roughly 15 to 50 US dollars** (third-party price lists give Gemini 3.5 Flash-Lite at $0.30 / $2.50 per million input / output tokens; not
checked against Google's page; reasoning tokens count as output). Nothing has been spent on a paid tier.

## Must do

| Item | Status | Evidence / note |
|---|---|---|
| Run the multi-turn benchmark (SnapRAG-LongConv) | **running, first variant only** (`full`, gemini-3.1-flash-lite, 80 turns); the other five variants wait for quota | the earlier attempt was lost: 33 of 80 turns failed on rate limits, so none of its numbers are used |
| Verification ladder (unsupported-claim rate, fact recall) | **ready; runs on the `full` conversation when it finishes** | `benchmarks/longconv/ladder.py`; needs about 200 more calls, with a stronger judge model that has its own small daily quota |
| Verifier error rate by hand labelling | **tool ready; labelling after the ladder run** | `benchmarks/verifier_audit.py`; one labeller (not independent) |
| Check novelty / verify the AB-RAG reference | **partly done** | AB-RAG verified (Kamthan, 27 June 2026); closest work to C1 found (Context Codec); "Cognis" not found and removed. Full texts and Google / Semantic Scholar still need a person. See "Novelty and citation check" |
| All 14 architectures on all 75 questions | **partly: 4 of 14 done** (`closed_book`, `dense_k5`, `bm25_k5`, `hybrid_rerank_k5` partly), then the daily quota of gemini-3.5-flash-lite ran out | `docqa_full75.json`; resumable (`--resume`) |

## Should do

| Item | Status | Note |
|---|---|---|
| Why recall fell from 90.9% to 79.0% with the real planner and HyDE | **ready, not run** | three diagnostic retrievals (`snaprag_fixed_k_noplan`, `snaprag_retrieval_planner`, `snaprag_retrieval`); about 5 calls a question |
| Calibrate the confidence | **ready, not run** | `snaprag_calib` keeps every attempt so any weights and threshold can be replayed offline; about 12 calls a question (900 calls) |
| Ablations fixed interval, fixed lambda, fixed k, no exemplars | **not run** | LongConv variants exist; about 440 calls each |
| A second document or domain, and a second LLM | **not done** | the two benchmarks use two different documents, but each benchmark has one; the only other free model that answered (gemini-3-flash-preview) allows 5 requests a minute |
| Repeat runs, seeds and dates | **seeds and dates recorded; no repeats** | each result file stores date, commit, seed, temperature |

## Evaluation fixes (all done)

Holm-Bonferroni over the paired tests; exact counts and Clopper-Pearson intervals instead of "100.0% (100-100)"; provider-reported tokens and seconds
without the rate-limit wait; `snaprag_fixed` renamed `snaprag_fixed_k`; "answers verified" renamed "passed the confidence check"; the 30-question sample is
documented in the report; LaTeX tables scale to the line width, have unique captions and labels. The 30-question pilot keeps its old numbers (no token counts) and is marked as a pilot.

## Writing and release

| Item | Status |
|---|---|
| Paper | draft in `paper/main.tex`: introduction, related work, setting, method with Algorithm 1, benchmarks, limitations. Abstract, results and conclusion are placeholders until the runs exist; the abstract is written last |
| Figures | architecture figure done; distance, ladder and architecture-comparison figures and the failure / kept-dropped examples are generated from result files by `paper/make_figures.py` once those exist |
| Appendix | generated from the code (prompts, probes, hyper-parameters, reproduction commands): `paper/build_appendix.py` |
| Licence check | *A Suitable Boy* (1993) is under copyright: release questions, answers and page numbers, not the PDF or page text. The 1917 torts textbook is probably public domain in the US; confirm for the scan used |
| Outside read / mock review / citation check / venue format | not done (needs people); `paper/self_review.md` is a self-review, not an outside read |


---

# SnapRAG-LongConv -- long-conversation results

Document: **cu31924019203078.pdf** | LLM: **gemini-3.1-flash-lite** | 80 turns, seed 0, 1 repeat(s) | 2026-10-23 17:40 | commit `24e59b2*`

Probes are scored by fixed patterns, not by an LLM judge. Each cell is k/n with an exact (Clopper-Pearson) 95% interval; a probe whose source turn failed to run is excluded. *Memory probes* are recall, supersede and reference; *memory loss* is the accuracy on near probes minus that on very-far probes.

## Probe accuracy

| Variant | All probes | Memory probes | recall | supersede | reference | absent | trap | near (memory) | mid (memory) | far (memory) | very_far (memory) | Memory loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full` | 17/24 = 71% (49-87) | 12/18 = 67% (41-87) | 10/12 = 83% (52-98) | 2/2 = 100% (16-100) | 0/4 = 0% (0-60) | 4/4 = 100% (40-100) | 1/2 = 50% (1-99) | 3/4 = 75% (19-99) | 4/5 = 80% (28-99) | 4/5 = 80% (28-99) | 1/4 = 25% (1-81) | +50 pp |

## Cost per turn (provider-reported)

Tokens and calls cover the answer and the memory upkeep of the turn. Active seconds exclude the wait for the request-rate limit (rows run before this was measured show a dash). Compression is the source tokens of a snapshot over its own tokens.

| Variant | Prompt tokens | Completion tokens | LLM calls (answer + upkeep) | Active seconds (mean / p95) | Doc page hit | Snapshots | Compression | State items | Errors |
|---|---|---|---|---|---|---|---|---|---|
| `full` | 34,662 | 1,390 | 5.4 | 115.4 / 107.8 | 100% | 19.0 | 1.9x | 20.0 | 1 |

## Why probes failed

- `full`: endorsed the false premise 1, forgot 6


---

# Document-QA architecture comparison -- answers

Document: **A Suitable Boy Vikram Seth.pdf** | LLM: **gemini-3.5-flash-lite** | 75 questions | run 2026-10-23 00:08 | commit `24e59b2*` | seed 0, temperature 0

## Architectures

| Name | What it is | Source |
|---|---|---|
| `closed_book` | no retrieval | baseline |
| `dense_k5` | dense top-5 | naive RAG |
| `bm25_k5` | BM25 top-5 | baseline |
| `hybrid_rerank_k5` | hybrid + cross-encoder top-5 | standard advanced RAG |
| `hybrid_rerank_k10` | hybrid + cross-encoder top-10 | standard advanced RAG, more pages |
| `hyde` | retrieve with a hypothetical answer | HyDE, Gao et al. 2022 (arXiv 2212.10496) |
| `snaprag` | SnapRAG: intent-sized batches + AB-RAG | this work |

Diagnostics (not competitors; used in the ablation and calibration sections):

| Name | What it is |
|---|---|
| `snaprag_retrieval` | SnapRAG retrieval only: LLM planner + HyDE, nothing answered |
| `snaprag_retrieval_planner` | SnapRAG retrieval only: LLM planner, no HyDE |

All architectures use the same index, embedder, cross-encoder, answer prompt and LLM. Questions: 6 aggregation, 16 detail, 7 false_premise, 14 inference, 23 multi_hop, 7 numeric, 2 unanswerable (9 unanswerable or false-premise). Scoring is by fixed patterns, not by an LLM judge.

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

- **Accuracy**: share of questions answered correctly, shown as k/n with an exact (Clopper-Pearson) 95% interval. An answerable question is correct when the answer contains every required fact (fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer declines or states the correction.
- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions; bootstrap interval).
- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).
- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).
- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.
- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.
- **LLM calls, tokens, pages read, seconds**: cost of one question. Tokens are the counts the provider reported (prompt and completion, all calls of the question); seconds are shown without the time spent waiting for the request-rate limit, and with it (wall).
- **Confidence check**: the AB-RAG checker's verdict on an answer (not the reconstruction check of the memory method); AUROC is the chance that a right answer has a higher confidence than a wrong one (0.5 = the confidence says nothing).
- **Comparisons** with SnapRAG are paired on the same questions: an exact two-sided sign test for right/wrong, a Wilcoxon signed-rank test for scores, Holm-Bonferroni adjustment across all 6 comparisons of each metric, and a percentile bootstrap interval (2000 resamples, seed 0) of the difference. A difference is called better or worse only when the adjusted p-value is below 0.05 and the interval excludes zero.

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
| `LLM_REQUESTS_PER_MINUTE` | 14 |
| `temperature` | 0 |

> **Partial results:** `bm25_k5`, `hybrid_rerank_k10`, `hyde`, `snaprag` have not answered every question (the run is unfinished or hit a rate limit), so their numbers rest on fewer questions.

## Summary

Accuracy ranking: `hybrid_rerank_k10` 100%; `snaprag` 74%; `hybrid_rerank_k5` 73%; `bm25_k5` 67%; `hyde` 67%; `dense_k5` 64%; `closed_book` 7%.

- SnapRAG against the best published method re-implemented here (`hyde`): 73.7% against 66.7%, +0.0 pp (-100 to +100); won 1, lost 1, tied 1; p=1.000, Holm 1.000; **no clear difference**.
- Against a standard hybrid + cross-encoder pipeline (`hybrid_rerank_k5`): 73.7% against 73.3%, -5.3 pp (-21 to +11); won 1, lost 2, tied 16; p=1.000, Holm 1.000; **no clear difference**.
- Cost: `snaprag` makes 3.7 LLM calls per question; `hyde` 2.0.

## Answer quality

| Architecture | Accuracy, all questions (exact 95% CI) | vs `dense_k5` | Answerable only | Mean fact score | Unanswerable handled | Refused an answerable question |
|---|---|---|---|---|---|---|
| `closed_book` | 5/75 = 6.7% (2-15) | -57.3 pp | 3/66 = 4.5% (1-13) | 10.5% (5-17) | 2/9 (22%) | 0/66 (0%) |
| `dense_k5` | 48/75 = 64.0% (52-75) | - | 39/66 = 59.1% (46-71) | 77.4% (70-85) | 9/9 (100%) | 3/66 (5%) |
| `bm25_k5` | 46/69 = 66.7% (54-78) | +2.7 pp | 42/65 = 64.6% (52-76) | 78.3% (70-86) | 4/4 (100%) | 6/65 (9%) |
| `hybrid_rerank_k5` | 55/75 = 73.3% (62-83) | +9.3 pp | 47/66 = 71.2% (59-82) | 84.3% (78-91) | 8/9 (89%) | 2/66 (3%) |
| `hybrid_rerank_k10` | **1/1 = 100.0% (2-100)** | +36.0 pp | 1/1 = 100.0% (2-100) | 1/1 = 100.0% (2-100) | - | 0/1 (0%) |
| `hyde` | 2/3 = 66.7% (9-99) | +2.7 pp | 2/3 = 66.7% (9-99) | 77.8% (33-100) | - | 0/3 (0%) |
| `snaprag` | 14/19 = 73.7% (49-91) | +9.7 pp | 14/19 = 73.7% (49-91) | 82.5% (67-96) | - | 2/19 (11%) |

## Retrieval quality

Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. These are measured on answerable questions only.

| Architecture | Context recall (95% CI) | Page hit (exact 95% CI) | Gold pages read | Pages read (mean) | Context (characters) |
|---|---|---|---|---|---|
| `dense_k5` | 81.2% (74-88) | 57/66 = 86.4% (76-94) | 69.1% | 5.0 | 6,432 |
| `bm25_k5` | 80.1% (72-88) | 58/65 = 89.2% (79-96) | 68.8% | 5.0 | 6,463 |
| `hybrid_rerank_k5` | 87.4% (81-93) | 63/66 = 95.5% (87-99) | 77.8% | 5.0 | 6,533 |
| `hybrid_rerank_k10` | **1/1 = 100.0% (2-100)** | 1/1 = 100.0% (2-100) | 100.0% | 10.0 | 16,159 |
| `hyde` | 77.8% (33-100) | 2/3 = 66.7% (9-99) | 66.7% | 5.0 | 5,789 |
| `snaprag` | 84.2% (70-96) | 18/19 = 94.7% (74-100) | 84.2% | 6.3 | 9,012 |

### Context recall by question type

| Architecture | detail (n=16) | numeric (n=7) | aggregation (n=6) | multi_hop (n=23) | inference (n=14) |
|---|---|---|---|---|---|
| `dense_k5` | 90.6% | 78.6% | 84.7% | 71.0% | 86.9% |
| `bm25_k5` | 83.3% | 54.8% | 68.1% | 82.6% | 90.5% |
| `hybrid_rerank_k5` | 100.0% | 69.0% | 88.9% | 79.0% | 95.2% |
| `hybrid_rerank_k10` | - | - | - | 100.0% | - |
| `hyde` | 100.0% | - | - | 33.3% | - |
| `snaprag` | 66.7% | 100.0% | 66.7% | 79.2% | 100.0% |

## Ablation: what the planner and HyDE do to retrieval

The same first batch of pages SnapRAG reads, retrieved three ways. Differences are paired against the first row (Wilcoxon signed-rank on context recall, Holm-adjusted).

| Retrieval | Context recall (95% CI) | Page hit | Gold pages read | Pages read (mean) | Difference in context recall vs first row |
|---|---|---|---|---|---|
| LLM planner, no HyDE | 90.9% (85-96) | 63/66 = 95.5% (87-99) | 81.6% | 5.0 | - |
| LLM planner + HyDE (SnapRAG's default) | 90.3% (79-100) | 12/12 = 100.0% (74-100) | 83.3% | 5.0 | +0.0 pp (+0 to +0); p=1.000, Holm 1.000 |

## Cost and speed

Tokens are the provider's own counts, summed over all calls of a question (rows run before token counting was added show a dash). Active seconds exclude the time the client waited for the request-rate limit; wall seconds include it, so they depend on the rate limit, not the method.

| Architecture | LLM calls | Prompt tokens | Completion tokens | Pages read | Accuracy per call | Active seconds (median / p95) | Wall seconds (median) | Errors |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 1.0 | 59 | 34 | 0.0 | 6.7 pp | 6.8 / 20.9 | 9.1 | 0 |
| `dense_k5` | 1.0 | 2,214 | 73 | 5.0 | 64.0 pp | 5.5 / 28.8 | 11.4 | 0 |
| `bm25_k5` | 1.0 | 2,213 | 77 | 5.0 | 66.7 pp | 3.6 / 28.8 | 5.2 | 0 |
| `hybrid_rerank_k5` | 1.0 | 2,223 | 74 | 5.0 | 73.3 pp | 3.3 / 6.8 | 4.7 | 0 |
| `hybrid_rerank_k10` | 1.0 | 4,672 | 91 | 10.0 | 100.0 pp | 21.3 / 21.3 | 30.3 | 0 |
| `hyde` | 2.0 | 2,087 | 74 | 5.0 | 33.3 pp | 7.3 / 16.9 | 7344.0 | 0 |
| `snaprag` | 3.7 | 7,631 | 255 | 6.3 | 20.0 pp | 16.9 / 31.7 | 63.2 | 1 |

## Confidence check

The AB-RAG checker judges whether the retrieved text supports the answer and gives a confidence; an answer below the stopping threshold triggers another batch of pages. A useful check separates right from wrong answers: accuracy should be higher among the answers that pass, and AUROC should be well above 0.5.

| Architecture | Mean confidence | Passed the check | Accuracy when passed | Accuracy when not passed | AUROC of confidence |
|---|---|---|---|---|---|
| `snaprag` | 0.86 | 19/19 (100%) | 14/19 = 73.7% (49-91) | - | 0.73 |

## Accuracy by question type

| Architecture | detail (n=16) | numeric (n=7) | aggregation (n=6) | multi_hop (n=23) | inference (n=14) | false_premise (n=7) | unanswerable (n=2) |
|---|---|---|---|---|---|---|---|
| `closed_book` | 1/16 (6%) | 1/7 (14%) | 0/6 (0%) | 0/23 (0%) | 1/14 (7%) | 2/7 (29%) | 0/2 (0%) |
| `dense_k5` | 14/16 (88%) | 4/7 (57%) | 3/6 (50%) | 10/23 (43%) | 8/14 (57%) | 7/7 (100%) | 2/2 (100%) |
| `bm25_k5` | 13/16 (81%) | 3/7 (43%) | 3/6 (50%) | 13/22 (59%) | 10/14 (71%) | 4/4 (100%) | - |
| `hybrid_rerank_k5` | 16/16 (100%) | 4/7 (57%) | 4/6 (67%) | 14/23 (61%) | 9/14 (64%) | 6/7 (86%) | 2/2 (100%) |
| `hybrid_rerank_k10` | - | - | - | 1/1 (100%) | - | - | - |
| `hyde` | 2/2 (100%) | - | - | 0/1 (0%) | - | - | - |
| `snaprag` | 2/3 (67%) | 3/3 (100%) | 0/1 (0%) | 5/8 (62%) | 4/4 (100%) | - | - |

## Accuracy by difficulty

| Architecture | easy (n=4) | medium (n=36) | hard (n=35) |
|---|---|---|---|
| `closed_book` | 1/4 (25%) | 2/36 (6%) | 2/35 (6%) |
| `dense_k5` | 4/4 (100%) | 24/36 (67%) | 20/35 (57%) |
| `bm25_k5` | 4/4 (100%) | 25/36 (69%) | 17/29 (59%) |
| `hybrid_rerank_k5` | 4/4 (100%) | 28/36 (78%) | 23/35 (66%) |
| `hybrid_rerank_k10` | - | 1/1 (100%) | - |
| `hyde` | - | 2/3 (67%) | - |
| `snaprag` | 2/2 (100%) | 8/11 (73%) | 4/6 (67%) |

## `snaprag` against every other architecture (paired, same questions)

Each cell: mean difference in percentage points (95% bootstrap interval); questions won, lost and tied; raw p-value and Holm-adjusted p-value over the 6 comparisons; the verdict.

| Compared with | Accuracy difference | Fact-score difference | Context-recall difference |
|---|---|---|---|
| `closed_book` | +57.9 pp (+37 to +79); won 11, lost 0, tied 8; p=0.001, Holm 0.006; **better** | +59.6 pp (+39 to +79); won 14, lost 1, tied 4; p=0.001, Holm 0.004; **better** | +84.2 pp (+70 to +96); won 18, lost 0, tied 1; p=0.000, Holm 0.000; **better** |
| `dense_k5` | +15.8 pp (-5 to +37); won 4, lost 1, tied 14; p=0.375, Holm 1.000; **no clear difference** | +2.6 pp (-12 to +17); won 4, lost 2, tied 13; p=0.462, Holm 1.000; **no clear difference** | +6.1 pp (-10 to +21); won 4, lost 1, tied 14; p=0.498, Holm 1.000; **no clear difference** |
| `bm25_k5` | +10.5 pp (+0 to +26); won 2, lost 0, tied 17; p=0.500, Holm 1.000; **no clear difference** | +9.6 pp (-5 to +25); won 4, lost 1, tied 14; p=0.279, Holm 1.000; **no clear difference** | +7.9 pp (-6 to +22); won 4, lost 1, tied 14; p=0.343, Holm 1.000; **no clear difference** |
| `hybrid_rerank_k5` | -5.3 pp (-21 to +11); won 1, lost 2, tied 16; p=1.000, Holm 1.000; **no clear difference** | -7.9 pp (-24 to +5); won 1, lost 2, tied 16; p=0.276, Holm 1.000; **no clear difference** | -6.1 pp (-20 to +5); won 1, lost 2, tied 16; p=0.285, Holm 1.000; **no clear difference** |
| `hybrid_rerank_k10` | +0.0 pp (+0 to +0); won 0, lost 0, tied 1; p=1.000, Holm 1.000; **no clear difference** | +0.0 pp (+0 to +0); won 0, lost 0, tied 1; p=1.000, Holm 1.000; **no clear difference** | +0.0 pp (+0 to +0); won 0, lost 0, tied 1; p=1.000, Holm 1.000; **no clear difference** |
| `hyde` | +0.0 pp (-100 to +100); won 1, lost 1, tied 1; p=1.000, Holm 1.000; **no clear difference** | -11.1 pp (-100 to +67); won 1, lost 1, tied 1; p=1.000, Holm 1.000; **no clear difference** | -11.1 pp (-100 to +67); won 1, lost 1, tied 1; p=1.000, Holm 1.000; **no clear difference** |

## Questions `snaprag` got wrong (5 of 19)

- **q05** (aggregation, medium): What was Dr Kishen Chand Seth's profession, and what was his diet restricted to?  
  expected: He was a radiologist (septuagenarian); his diet was boiled eggs, weak tea, lemon squash and arrowroot biscuits.  
  answered: The text states: * **Profession:** Dr Kishen Chand Seth was the Principal of a medical college, from which he had retired (PDF p. 70). * **Diet restrictions:** Owing to a digestive disorder combined with diabetes, his diet was confined to boiled eggs, weak tea
- **q06** (multi_hop, medium): Where does Arun Mehra work, what is his brother Varun studying, and where do the two of them live?  
  expected: Arun is an executive at Bentsen & Pryce in Calcutta; Varun studies mathematics at Calcutta University and lives with Arun and Meenakshi in Calcutta.  
  answered: The text states: - Arun works at an office (he leaves for the office, and Basil Cox is coming for dinner as the "boss of her darling son" [Source: A Suitable Boy Vikram Seth.pdf, PDF pages 381, 385]). - Varun is studying mathematics at Calcutta University ([So
- **q13** (detail, medium): Who saved the convoy Kedarnath was travelling in when it was attacked near the Ravi Bridge?  
  expected: A Gurkha regiment coming from the other side opened fire and the attackers fled.  
  answered: Insufficient information found.
- **q19** (multi_hop, hard): What match did Dr Seth suggest for Lata, and what did he say about a possible double wedding involving Maan?  
  expected: He suggested a young radiologist at the Prince of Wales; he said they could have captured Maan for a double wedding, but Maan was now said to be engaged to a Banaras girl.  
  answered: Insufficient information found.
- **q20** (multi_hop, hard): What happened when Mrs Rupa Mehra and her four children went to live with her father after her husband's death?  
  expected: It lasted only a few months and was a disaster; Seth accused her of making eyes at his colleagues, Arun threatened to beat him up, and she left with the children for Darjeeling.  
  answered: The text states: - Mrs Rupa Mehra and her family went to live with her father after her husband's death because of her straitened finances, thinking he might be lonely, and hoping to help him with her household affairs (PDF p. 69, printed p. 42-). - The experi

## Notes

- One document, one LLM and 75 questions written by reading the book: the intervals show how little this many questions can separate close systems. Treat differences that are not marked better or worse as noise.
- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.
- The book is a scanned novel; some pages have broken text, which lowers every architecture equally. It is under copyright, so neither the PDF nor its page text is released; the questions, answers and gold page numbers are.
- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.
- Scoring requires every fact in the answer, so an answer that is right but leaves one fact out counts as wrong; the fact score shows the partial credit.


---

# Document-QA architecture comparison -- retrieval only (no LLM answers)

Document: **A Suitable Boy Vikram Seth.pdf** | LLM: **none** | 75 questions | run 2026-10-22 10:45

## Architectures

| Name | What it is | Source |
|---|---|---|
| `dense_k5` | dense top-5 | naive RAG |
| `bm25_k5` | BM25 top-5 | baseline |
| `hybrid_k5` | dense + BM25 (RRF) top-5 | standard hybrid RAG |
| `hybrid_rerank_k5` | hybrid + cross-encoder top-5 | standard advanced RAG |
| `hybrid_rerank_k10` | hybrid + cross-encoder top-10 | standard advanced RAG, more pages |
| `snaprag_fixed_k` | SnapRAG retrieval + verification, AB-RAG off | this work |

All architectures use the same index, embedder and cross-encoder; no LLM answers are generated, and SnapRAG's planner and HyDE run their offline fallbacks. Questions: 6 aggregation, 16 detail, 7 false_premise, 14 inference, 23 multi_hop, 7 numeric, 2 unanswerable (9 unanswerable or false-premise). Scoring is by fixed patterns, not by an LLM judge.

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

- **Accuracy**: share of questions answered correctly, shown as k/n with an exact (Clopper-Pearson) 95% interval. An answerable question is correct when the answer contains every required fact (fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer declines or states the correction.
- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions; bootstrap interval).
- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).
- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).
- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.
- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.
- **LLM calls, tokens, pages read, seconds**: cost of one question. Tokens are the counts the provider reported (prompt and completion, all calls of the question); seconds are shown without the time spent waiting for the request-rate limit, and with it (wall).
- **Confidence check**: the AB-RAG checker's verdict on an answer (not the reconstruction check of the memory method); AUROC is the chance that a right answer has a higher confidence than a wrong one (0.5 = the confidence says nothing).
- **Comparisons** with SnapRAG are paired on the same questions: an exact two-sided sign test for right/wrong, a Wilcoxon signed-rank test for scores, Holm-Bonferroni adjustment across all 5 comparisons of each metric, and a percentile bootstrap interval (2000 resamples, seed 0) of the difference. A difference is called better or worse only when the adjusted p-value is below 0.05 and the interval excludes zero.

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
| `LLM_REQUESTS_PER_MINUTE` | 14 |
| `temperature` | 0 |

## Retrieval quality

Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. These are measured on answerable questions only.

| Architecture | Context recall (95% CI) | Page hit (exact 95% CI) | Gold pages read | Pages read (mean) | Context (characters) |
|---|---|---|---|---|---|
| `dense_k5` | 81.2% (74-88) | 57/66 = 86.4% (76-94) | 69.1% | 5.0 | 6,432 |
| `bm25_k5` | 80.4% (72-88) | 59/66 = 89.4% (79-96) | 68.8% | 5.0 | 6,429 |
| `hybrid_k5` | 84.0% (77-91) | 60/66 = 90.9% (81-97) | 73.4% | 5.0 | 6,314 |
| `hybrid_rerank_k5` | 87.4% (81-93) | 63/66 = 95.5% (87-99) | 77.8% | 5.0 | 6,533 |
| `hybrid_rerank_k10` | **91.4% (86-96)** | 64/66 = 97.0% (89-100) | 85.2% | 10.0 | 12,056 |
| `snaprag_fixed_k` | 90.9% (85-96) | 63/66 = 95.5% (87-99) | 81.6% | 5.0 | 7,134 |

### Context recall by question type

| Architecture | detail (n=16) | numeric (n=7) | aggregation (n=6) | multi_hop (n=23) | inference (n=14) |
|---|---|---|---|---|---|
| `dense_k5` | 90.6% | 78.6% | 84.7% | 71.0% | 86.9% |
| `bm25_k5` | 83.3% | 54.8% | 68.1% | 83.3% | 90.5% |
| `hybrid_k5` | 87.5% | 69.0% | 90.3% | 79.0% | 92.9% |
| `hybrid_rerank_k5` | 100.0% | 69.0% | 88.9% | 79.0% | 95.2% |
| `hybrid_rerank_k10` | 100.0% | 85.7% | 94.4% | 84.1% | 95.2% |
| `snaprag_fixed_k` | 100.0% | 85.7% | 88.9% | 84.1% | 95.2% |

## `snaprag_fixed_k` against every other architecture (paired, same questions; retrieval only)

| Compared with | Context-recall difference |
|---|---|
| `dense_k5` | +9.7 pp (+4 to +16); won 13, lost 1, tied 52; p=0.006, Holm 0.030; **better** |
| `bm25_k5` | +10.5 pp (+3 to +19); won 12, lost 3, tied 51; p=0.013, Holm 0.052; **no clear difference** |
| `hybrid_k5` | +6.9 pp (+1 to +13); won 10, lost 2, tied 54; p=0.034, Holm 0.101; **no clear difference** |
| `hybrid_rerank_k5` | +3.5 pp (+1 to +7); won 4, lost 0, tied 62; p=0.063, Holm 0.127; **no clear difference** |
| `hybrid_rerank_k10` | -0.5 pp (-4 to +3); won 1, lost 2, tied 63; p=1.000, Holm 1.000; **no clear difference** |

## Notes

- One document, one LLM and 75 questions written by reading the book: the intervals show how little this many questions can separate close systems. Treat differences that are not marked better or worse as noise.
- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.
- The book is a scanned novel; some pages have broken text, which lowers every architecture equally. It is under copyright, so neither the PDF nor its page text is released; the questions, answers and gold page numbers are.
- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.
- Scoring requires every fact in the answer, so an answer that is right but leaves one fact out counts as wrong; the fact score shows the partial credit.


---

# Pilot: the first 30-question comparison (before token counting, calibration and the full run)

Document: **A Suitable Boy Vikram Seth.pdf** | LLM: **gemini-3.5-flash-lite** | 30 questions | run 2026-10-22 11:12

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
| `snaprag_fixed_k` | SnapRAG retrieval + verification, AB-RAG off | this work |
| `snaprag` | SnapRAG: intent-sized batches + AB-RAG | this work |

All architectures use the same index, embedder, cross-encoder, answer prompt and LLM. Questions: 3 aggregation, 6 detail, 3 false_premise, 5 inference, 10 multi_hop, 3 numeric (3 unanswerable or false-premise). Scoring is by fixed patterns, not by an LLM judge.

## Dataset

30 questions written by reading the book; every answerable question lists the facts an answer needs and the gold pages that hold them (57 distinct pages, from page 9 to 428). Multi-hop questions need facts from two or more pages that are not neighbours.
This is an evenly spaced sample of 30 of the dataset's 75 questions (question i of the sample is question floor(i x 2.50) of the dataset, `--sample 30`). It was chosen by position, not by any result, only to fit the free-tier request limit, and is the same for every architecture.

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

- **Accuracy**: share of questions answered correctly, shown as k/n with an exact (Clopper-Pearson) 95% interval. An answerable question is correct when the answer contains every required fact (fixed patterns, no LLM judge); an unanswerable or false-premise question is correct when the answer declines or states the correction.
- **Fact score**: mean share of required facts found in the answer (partial credit, answerable questions; bootstrap interval).
- **Unanswerable handled**: accuracy on the unanswerable and false-premise questions (does the system refuse instead of inventing?).
- **Refused an answerable question**: share of answerable questions on which the answer is a refusal (over-caution).
- **Context recall**: share of required facts that appear in the pages the LLM read. It measures retrieval, independent of the answer.
- **Page hit** and **gold pages read**: whether a gold page was read, and the share of gold pages read.
- **LLM calls, tokens, pages read, seconds**: cost of one question. Tokens are the counts the provider reported (prompt and completion, all calls of the question); seconds are shown without the time spent waiting for the request-rate limit, and with it (wall).
- **Confidence check**: the AB-RAG checker's verdict on an answer (not the reconstruction check of the memory method); AUROC is the chance that a right answer has a higher confidence than a wrong one (0.5 = the confidence says nothing).
- **Comparisons** with SnapRAG are paired on the same questions: an exact two-sided sign test for right/wrong, a Wilcoxon signed-rank test for scores, Holm-Bonferroni adjustment across all 13 comparisons of each metric, and a percentile bootstrap interval (2000 resamples, seed 0) of the difference. A difference is called better or worse only when the adjusted p-value is below 0.05 and the interval excludes zero.

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
| `LLM_REQUESTS_PER_MINUTE` | 14 |
| `temperature` | 0 |

## Summary

Accuracy ranking: `ircot` 70%; `abrag` 70%; `snaprag` 70%; `hybrid_rerank_k5` 67%; `hybrid_rerank_k10` 67%; `dense_k5` 63%; `hybrid_k5` 63%; `adaptive_rag` 63%; `snaprag_fixed_k` 63%; `crag` 57%; `bm25_k5` 53%; `hyde` 53%; `self_rag` 43%; `closed_book` 7%.

- SnapRAG against the best published method re-implemented here (`ircot`): 70.0% against 70.0%, +0.0 pp (-17 to +17); won 3, lost 3, tied 24; p=1.000, Holm 1.000; **no clear difference**.
- Against SnapRAG without the extra AB-RAG batches (`snaprag_fixed_k`), i.e. the effect of the adaptive batches: 70.0% against 63.3%, +6.7 pp (-13 to +23); won 5, lost 3, tied 22; p=0.727, Holm 1.000; **no clear difference**.
- Against AB-RAG with one fixed first batch and budget for every question (`abrag`), i.e. the effect of the intent-aware batch: 70.0% against 70.0%, +0.0 pp (-20 to +20); won 4, lost 4, tied 22; p=1.000, Holm 1.000; **no clear difference**.
- Against a standard hybrid + cross-encoder pipeline (`hybrid_rerank_k5`): 70.0% against 66.7%, +3.3 pp (-13 to +20); won 4, lost 3, tied 23; p=1.000, Holm 1.000; **no clear difference**.
- Cost: `snaprag` makes 3.7 LLM calls per question; `hyde` 2.0, `crag` 2.3, `self_rag` 4.0, `ircot` 3.3, `adaptive_rag` 3.0, `abrag` 3.5.

## Answer quality

| Architecture | Accuracy, all questions (exact 95% CI) | vs `dense_k5` | Answerable only | Mean fact score | Unanswerable handled | Refused an answerable question |
|---|---|---|---|---|---|---|
| `closed_book` | 2/30 = 6.7% (1-22) | -56.7 pp | 2/27 = 7.4% (1-24) | 14.2% (5-25) | 0/3 (0%) | 0/27 (0%) |
| `dense_k5` | 19/30 = 63.3% (44-80) | - | 16/27 = 59.3% (39-78) | 79.6% (68-90) | 3/3 (100%) | 1/27 (4%) |
| `bm25_k5` | 16/30 = 53.3% (34-72) | -10.0 pp | 13/27 = 48.1% (29-68) | 71.0% (58-84) | 3/3 (100%) | 3/27 (11%) |
| `hybrid_k5` | 19/30 = 63.3% (44-80) | +0.0 pp | 16/27 = 59.3% (39-78) | 75.9% (63-88) | 3/3 (100%) | 2/27 (7%) |
| `hybrid_rerank_k5` | 20/30 = 66.7% (47-83) | +3.3 pp | 17/27 = 63.0% (42-81) | 79.0% (66-91) | 3/3 (100%) | 3/27 (11%) |
| `hybrid_rerank_k10` | 20/30 = 66.7% (47-83) | +3.3 pp | 17/27 = 63.0% (42-81) | 82.1% (71-93) | 3/3 (100%) | 1/27 (4%) |
| `hyde` | 16/30 = 53.3% (34-72) | -10.0 pp | 13/27 = 48.1% (29-68) | 66.0% (51-80) | 3/3 (100%) | 4/27 (15%) |
| `crag` | 17/30 = 56.7% (37-75) | -6.7 pp | 14/27 = 51.9% (32-71) | 74.1% (62-85) | 3/3 (100%) | 2/27 (7%) |
| `self_rag` | 13/30 = 43.3% (25-63) | -20.0 pp | 10/27 = 37.0% (19-58) | 62.3% (48-76) | 3/3 (100%) | 6/27 (22%) |
| `ircot` | **21/30 = 70.0% (51-85)** | +6.7 pp | 18/27 = 66.7% (46-83) | 85.2% (75-94) | 3/3 (100%) | 1/27 (4%) |
| `adaptive_rag` | 19/30 = 63.3% (44-80) | +0.0 pp | 16/27 = 59.3% (39-78) | 77.8% (65-90) | 3/3 (100%) | 3/27 (11%) |
| `abrag` | **21/30 = 70.0% (51-85)** | +6.7 pp | 18/27 = 66.7% (46-83) | 82.7% (71-93) | 3/3 (100%) | 2/27 (7%) |
| `snaprag_fixed_k` | 19/30 = 63.3% (44-80) | +0.0 pp | 16/27 = 59.3% (39-78) | 77.2% (64-88) | 3/3 (100%) | 2/27 (7%) |
| `snaprag` | **21/30 = 70.0% (51-85)** | +6.7 pp | 18/27 = 66.7% (46-83) | 83.3% (72-93) | 3/3 (100%) | 2/27 (7%) |

## Retrieval quality

Context recall = share of required facts present in the pages the LLM read. Page hit = a gold page was read. These are measured on answerable questions only.

| Architecture | Context recall (95% CI) | Page hit (exact 95% CI) | Gold pages read | Pages read (mean) | Context (characters) |
|---|---|---|---|---|---|
| `dense_k5` | 80.9% (70-91) | 24/27 = 88.9% (71-98) | 67.3% | 5.0 | 6,632 |
| `bm25_k5` | 72.2% (58-85) | 24/27 = 88.9% (71-98) | 64.2% | 5.0 | 6,623 |
| `hybrid_k5` | 78.4% (66-90) | 24/27 = 88.9% (71-98) | 72.2% | 5.0 | 6,667 |
| `hybrid_rerank_k5` | 81.5% (69-93) | 24/27 = 88.9% (71-98) | 72.8% | 5.0 | 6,690 |
| `hybrid_rerank_k10` | 85.8% (74-95) | 25/27 = 92.6% (76-99) | 80.2% | 10.0 | 12,340 |
| `hyde` | 67.9% (52-82) | 23/27 = 85.2% (66-96) | 61.7% | 5.0 | 6,833 |
| `crag` | 77.8% (65-90) | 25/27 = 92.6% (76-99) | 60.5% | 2.4 | 3,547 |
| `self_rag` | 74.1% (59-87) | 23/27 = 85.2% (66-96) | 67.3% | 3.0 | 4,475 |
| `ircot` | **88.9% (79-97)** | 26/27 = 96.3% (81-100) | 77.8% | 4.3 | 6,374 |
| `adaptive_rag` | 81.5% (69-93) | 24/27 = 88.9% (71-98) | 74.1% | 4.5 | 6,207 |
| `abrag` | 85.2% (73-95) | 25/27 = 92.6% (76-99) | 76.5% | 7.0 | 10,002 |
| `snaprag_fixed_k` | 79.0% (65-90) | 25/27 = 92.6% (76-99) | 72.2% | 5.4 | 8,016 |
| `snaprag` | 85.8% (74-95) | 26/27 = 96.3% (81-100) | 77.2% | 6.2 | 9,153 |

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
| `snaprag_fixed_k` | 83.3% | 33.3% | 88.9% | 80.0% | 93.3% |
| `snaprag` | 83.3% | 66.7% | 100.0% | 85.0% | 93.3% |

## Cost and speed

Tokens are the provider's own counts, summed over all calls of a question (rows run before token counting was added show a dash). Active seconds exclude the time the client waited for the request-rate limit; wall seconds include it, so they depend on the rate limit, not the method.

| Architecture | LLM calls | Prompt tokens | Completion tokens | Pages read | Accuracy per call | Active seconds (median / p95) | Wall seconds (median) | Errors |
|---|---|---|---|---|---|---|---|---|
| `closed_book` | 1.0 | - | - | 0.0 | 6.7 pp | - | 4.2 | 0 |
| `dense_k5` | 1.0 | - | - | 5.0 | 63.3 pp | - | 3.5 | 0 |
| `bm25_k5` | 1.0 | - | - | 5.0 | 53.3 pp | - | 9.7 | 0 |
| `hybrid_k5` | 1.0 | - | - | 5.0 | 63.3 pp | - | 12.0 | 0 |
| `hybrid_rerank_k5` | 1.0 | - | - | 5.0 | 66.7 pp | - | 11.3 | 0 |
| `hybrid_rerank_k10` | 1.0 | - | - | 10.0 | 66.7 pp | - | 12.5 | 0 |
| `hyde` | 2.0 | - | - | 5.0 | 26.7 pp | - | 33.6 | 0 |
| `crag` | 2.3 | - | - | 2.4 | 25.0 pp | - | 52.4 | 0 |
| `self_rag` | 4.0 | - | - | 3.0 | 10.8 pp | - | 67.2 | 0 |
| `ircot` | 3.3 | - | - | 4.3 | 21.4 pp | - | 53.1 | 0 |
| `adaptive_rag` | 3.0 | - | - | 4.5 | 21.1 pp | - | 56.0 | 0 |
| `abrag` | 3.5 | - | - | 7.0 | 20.2 pp | - | 67.2 | 0 |
| `snaprag_fixed_k` | 3.1 | - | - | 5.4 | 20.7 pp | - | 35.1 | 0 |
| `snaprag` | 3.7 | - | - | 6.2 | 18.9 pp | - | 59.5 | 0 |

## Confidence check

The AB-RAG checker judges whether the retrieved text supports the answer and gives a confidence; an answer below the stopping threshold triggers another batch of pages. A useful check separates right from wrong answers: accuracy should be higher among the answers that pass, and AUROC should be well above 0.5.

| Architecture | Mean confidence | Passed the check | Accuracy when passed | Accuracy when not passed | AUROC of confidence |
|---|---|---|---|---|---|
| `abrag` | 0.86 | 30/30 (100%) | 21/30 = 70.0% (51-85) | - | 0.64 |
| `snaprag_fixed_k` | 0.86 | 29/30 (97%) | 19/29 = 65.5% (46-82) | 0/1 = 0.0% (0-98) | 0.53 |
| `snaprag` | 0.88 | 30/30 (100%) | 21/30 = 70.0% (51-85) | - | 0.48 |

`adaptive_rag` routed 0 questions to no retrieval, 17 to single-step retrieval and 13 to multi-step retrieval.

## Accuracy by question type

| Architecture | detail (n=6) | numeric (n=3) | aggregation (n=3) | multi_hop (n=10) | inference (n=5) | false_premise (n=3) |
|---|---|---|---|---|---|---|
| `closed_book` | 0/6 (0%) | 1/3 (33%) | 0/3 (0%) | 0/10 (0%) | 1/5 (20%) | 0/3 (0%) |
| `dense_k5` | 6/6 (100%) | 1/3 (33%) | 2/3 (67%) | 4/10 (40%) | 3/5 (60%) | 3/3 (100%) |
| `bm25_k5` | 3/6 (50%) | 1/3 (33%) | 2/3 (67%) | 3/10 (30%) | 4/5 (80%) | 3/3 (100%) |
| `hybrid_k5` | 5/6 (83%) | 1/3 (33%) | 3/3 (100%) | 3/10 (30%) | 4/5 (80%) | 3/3 (100%) |
| `hybrid_rerank_k5` | 6/6 (100%) | 1/3 (33%) | 2/3 (67%) | 4/10 (40%) | 4/5 (80%) | 3/3 (100%) |
| `hybrid_rerank_k10` | 6/6 (100%) | 2/3 (67%) | 2/3 (67%) | 3/10 (30%) | 4/5 (80%) | 3/3 (100%) |
| `hyde` | 4/6 (67%) | 1/3 (33%) | 3/3 (100%) | 3/10 (30%) | 2/5 (40%) | 3/3 (100%) |
| `crag` | 6/6 (100%) | 1/3 (33%) | 2/3 (67%) | 3/10 (30%) | 2/5 (40%) | 3/3 (100%) |
| `self_rag` | 3/6 (50%) | 1/3 (33%) | 2/3 (67%) | 1/10 (10%) | 3/5 (60%) | 3/3 (100%) |
| `ircot` | 6/6 (100%) | 1/3 (33%) | 3/3 (100%) | 4/10 (40%) | 4/5 (80%) | 3/3 (100%) |
| `adaptive_rag` | 6/6 (100%) | 1/3 (33%) | 2/3 (67%) | 3/10 (30%) | 4/5 (80%) | 3/3 (100%) |
| `abrag` | 6/6 (100%) | 2/3 (67%) | 2/3 (67%) | 4/10 (40%) | 4/5 (80%) | 3/3 (100%) |
| `snaprag_fixed_k` | 5/6 (83%) | 1/3 (33%) | 2/3 (67%) | 5/10 (50%) | 3/5 (60%) | 3/3 (100%) |
| `snaprag` | 5/6 (83%) | 2/3 (67%) | 3/3 (100%) | 5/10 (50%) | 3/5 (60%) | 3/3 (100%) |

## Accuracy by difficulty

| Architecture | easy (n=3) | medium (n=14) | hard (n=13) |
|---|---|---|---|
| `closed_book` | 1/3 (33%) | 1/14 (7%) | 0/13 (0%) |
| `dense_k5` | 3/3 (100%) | 9/14 (64%) | 7/13 (54%) |
| `bm25_k5` | 3/3 (100%) | 5/14 (36%) | 8/13 (62%) |
| `hybrid_k5` | 3/3 (100%) | 8/14 (57%) | 8/13 (62%) |
| `hybrid_rerank_k5` | 3/3 (100%) | 9/14 (64%) | 8/13 (62%) |
| `hybrid_rerank_k10` | 3/3 (100%) | 10/14 (71%) | 7/13 (54%) |
| `hyde` | 3/3 (100%) | 8/14 (57%) | 5/13 (38%) |
| `crag` | 3/3 (100%) | 8/14 (57%) | 6/13 (46%) |
| `self_rag` | 3/3 (100%) | 5/14 (36%) | 5/13 (38%) |
| `ircot` | 3/3 (100%) | 10/14 (71%) | 8/13 (62%) |
| `adaptive_rag` | 3/3 (100%) | 9/14 (64%) | 7/13 (54%) |
| `abrag` | 3/3 (100%) | 10/14 (71%) | 8/13 (62%) |
| `snaprag_fixed_k` | 2/3 (67%) | 9/14 (64%) | 8/13 (62%) |
| `snaprag` | 3/3 (100%) | 11/14 (79%) | 7/13 (54%) |

## `snaprag` against every other architecture (paired, same questions)

Each cell: mean difference in percentage points (95% bootstrap interval); questions won, lost and tied; raw p-value and Holm-adjusted p-value over the 13 comparisons; the verdict.

| Compared with | Accuracy difference | Fact-score difference | Context-recall difference |
|---|---|---|---|
| `closed_book` | +63.3 pp (+47 to +80); won 19, lost 0, tied 11; p=0.000, Holm 0.000; **better** | +72.2 pp (+58 to +86); won 26, lost 0, tied 4; p=0.000, Holm 0.000; **better** | +85.8 pp (+74 to +95); won 25, lost 0, tied 2; p=0.000, Holm 0.000; **better** |
| `dense_k5` | +6.7 pp (-10 to +27); won 5, lost 3, tied 22; p=0.727, Holm 1.000; **no clear difference** | +3.3 pp (-8 to +15); won 5, lost 3, tied 22; p=0.522, Holm 1.000; **no clear difference** | +4.9 pp (-9 to +19); won 6, lost 3, tied 18; p=0.435, Holm 1.000; **no clear difference** |
| `bm25_k5` | +16.7 pp (+0 to +37); won 7, lost 2, tied 21; p=0.180, Holm 1.000; **no clear difference** | +11.1 pp (+1 to +23); won 7, lost 2, tied 21; p=0.072, Holm 0.717; **no clear difference** | +13.6 pp (+2 to +27); won 6, lost 1, tied 20; p=0.047, Holm 0.521; **no clear difference** |
| `hybrid_k5` | +6.7 pp (-10 to +23); won 4, lost 2, tied 24; p=0.688, Holm 1.000; **no clear difference** | +6.7 pp (-2 to +17); won 5, lost 2, tied 23; p=0.197, Holm 1.000; **no clear difference** | +7.4 pp (-2 to +19); won 4, lost 1, tied 22; p=0.157, Holm 1.000; **no clear difference** |
| `hybrid_rerank_k5` | +3.3 pp (-13 to +20); won 4, lost 3, tied 23; p=1.000, Holm 1.000; **no clear difference** | +3.9 pp (-8 to +16); won 5, lost 3, tied 22; p=0.481, Holm 1.000; **no clear difference** | +4.3 pp (-9 to +17); won 5, lost 3, tied 19; p=0.481, Holm 1.000; **no clear difference** |
| `hybrid_rerank_k10` | +3.3 pp (-13 to +20); won 4, lost 3, tied 23; p=1.000, Holm 1.000; **no clear difference** | +1.1 pp (-11 to +12); won 4, lost 3, tied 23; p=0.864, Holm 1.000; **no clear difference** | +0.0 pp (-12 to +12); won 3, lost 3, tied 21; p=1.000, Holm 1.000; **no clear difference** |
| `hyde` | +16.7 pp (+0 to +33); won 6, lost 1, tied 23; p=0.125, Holm 1.000; **no clear difference** | +15.6 pp (+5 to +26); won 9, lost 1, tied 20; p=0.010, Holm 0.113; **no clear difference** | +17.9 pp (+7 to +30); won 8, lost 1, tied 18; p=0.010, Holm 0.124; **no clear difference** |
| `crag` | +13.3 pp (-7 to +33); won 7, lost 3, tied 20; p=0.344, Holm 1.000; **no clear difference** | +8.3 pp (-4 to +21); won 8, lost 3, tied 19; p=0.164, Holm 1.000; **no clear difference** | +8.0 pp (-6 to +20); won 8, lost 3, tied 16; p=0.246, Holm 1.000; **no clear difference** |
| `self_rag` | +26.7 pp (+10 to +47); won 9, lost 1, tied 20; p=0.021, Holm 0.258; **no clear difference** | +18.9 pp (+8 to +32); won 10, lost 1, tied 19; p=0.007, Holm 0.083; **no clear difference** | +11.7 pp (+0 to +25); won 6, lost 2, tied 19; p=0.078, Holm 0.778; **no clear difference** |
| `ircot` | +0.0 pp (-17 to +17); won 3, lost 3, tied 24; p=1.000, Holm 1.000; **no clear difference** | -1.7 pp (-12 to +7); won 3, lost 3, tied 24; p=0.832, Holm 1.000; **no clear difference** | -3.1 pp (-14 to +6); won 2, lost 3, tied 22; p=0.680, Holm 1.000; **no clear difference** |
| `adaptive_rag` | +6.7 pp (-13 to +23); won 5, lost 3, tied 22; p=0.727, Holm 1.000; **no clear difference** | +5.0 pp (-7 to +17); won 6, lost 3, tied 21; p=0.403, Holm 1.000; **no clear difference** | +4.3 pp (-9 to +17); won 5, lost 3, tied 19; p=0.481, Holm 1.000; **no clear difference** |
| `abrag` | +0.0 pp (-20 to +20); won 4, lost 4, tied 22; p=1.000, Holm 1.000; **no clear difference** | +0.6 pp (-11 to +11); won 5, lost 4, tied 21; p=0.764, Holm 1.000; **no clear difference** | +0.6 pp (-12 to +12); won 4, lost 3, tied 20; p=0.798, Holm 1.000; **no clear difference** |
| `snaprag_fixed_k` | +6.7 pp (-13 to +23); won 5, lost 3, tied 22; p=0.727, Holm 1.000; **no clear difference** | +5.6 pp (-4 to +16); won 5, lost 3, tied 22; p=0.226, Holm 1.000; **no clear difference** | +6.8 pp (-2 to +18); won 4, lost 1, tied 22; p=0.216, Holm 1.000; **no clear difference** |

## Questions `snaprag` got wrong (9 of 30)

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

- One document, one LLM and 30 questions written by reading the book: the intervals show how little this many questions can separate close systems. Treat differences that are not marked better or worse as noise.
- The published methods are simplified re-implementations from the papers' descriptions, run on the same small LLM as everything else; they are not the authors' trained models or code, so a result here does not reproduce a paper's own numbers.
- The book is a scanned novel; some pages have broken text, which lowers every architecture equally. It is under copyright, so neither the PDF nor its page text is released; the questions, answers and gold page numbers are.
- `closed_book` shows what the LLM already knows about this famous novel; a retrieval system should beat it on details and on the unanswerable questions.
- Scoring requires every fact in the answer, so an answer that is right but leaves one fact out counts as wrong; the fact score shows the partial credit.


---

# Novelty and citation check (searched 2026-10-23, web search and arXiv abstract pages only)

What was checked, what was found, and what is still open. Only abstracts were read, so every "not found" is a statement about
this limited search, not a proof that nothing exists.

| Item | Result | Consequence for the paper |
|---|---|---|
| AB-RAG, arXiv 2606.29090 | **Exists.** "AB-RAG: Adaptive Budgeted Retrieval-Augmented Generation for Reliable Question Answering", Ansh Kamthan, 27 June 2026, cs.CL. Training-free, backbone-agnostic: answer, estimate confidence, stop or retrieve more within a budget. Confidence = the model's own certainty (token probabilities where exposed, **self-consistency for closed APIs**), answer-evidence agreement, variance of retrieval scores. It also reports negative findings (a signal that worked poorly for short answers). | Cite it (done). Our S1 is a *judged* confidence, not token probabilities and not self-consistency: state this as a deviation (done). The abstract gives no formula or weights, so our 0.50/0.25/0.25 are ours; read the full paper before describing its equations. |
| Context Codec, arXiv 2605.17304 (Trukhina, Vashkelis, 17 May 2026) | **Closest to C1.** Compresses chat history into typed, source-grounded atoms with evidence spans, with a verification stage; metrics: critical-atom recall, round-trip recoverability. The abstract does not describe per-sentence verification of reconstructed text against an archive. | Cited and contrasted (done). C1's claim must stay narrow: sentence-level verification of text *reconstructed from a lossy snapshot* against the snapshot, the anchored document pages and an immutable archive. Read the full paper before submitting; if it already does this, C1 is not novel. |
| LCM, arXiv 2605.04050 | **Exists.** Immutable store plus summary nodes that can be expanded into the original; the store is the source of truth. | Cited. It does not check reconstructed text, per the abstract and write-ups. Author list still to be filled in refs.bib. |
| MemCoRe, arXiv 2602.07885 | **Exists.** Recovering evidence from progressively compressed facts; evaluated on LoCoMo (F1, BLEU-1), graph-backed. | Cited. |
| ChronoMem, arXiv 2607.27773 | **Exists** (Su, Xu, Zuo, Bertino, 30 July 2026): version control and rollback for agent memory. | Cited. |
| MemRefine, arXiv 2606.13177 | **Exists:** LLM-guided compression of an already built long-term memory. | Cited. |
| Other related papers found | CMV (arXiv 2602.22402, DAG-based structurally lossless trimming), ACM (2607.23809), Addressable Recall Compaction (2607.25066). Not read. | Read their abstracts before the final related-work section. |
| **Cognis, arXiv 2604.19771** | **Not found** by search (no paper under that ID or name). | Removed from the references. Either find the correct ID or drop it. |
| MTRAG, arXiv 2501.03468 | The search returned nothing for it; the ID is from memory. | Check on arXiv before citing. |
| "No long document-QA conversation benchmark with page citations" (C3) | Two searches found LoCoMo, LongMemEval and vendor memory benchmarks, none with documents and page-cited answers. | Keep the wording "we know of none" and the limited-search caveat. Search Semantic Scholar and Google Scholar too. |

Still to do by a person: a search on Google Scholar and Semantic Scholar ("verified reconstruction", "memory hallucination",
"context quality score", "intent-aware adaptive retrieval"), reading the full texts of AB-RAG and Context Codec, and a second
reader of the related-work table.
