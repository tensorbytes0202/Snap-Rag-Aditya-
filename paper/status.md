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
