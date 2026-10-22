# SnapRAG: paper template (10 pages, page by page)

How to use: each page has (1) what goes there, (2) the facts from this project, (3) a text skeleton.
`[ ]` = number or sentence to fill only after the benchmark has run. **Never write a result that has not been measured.**
Format assumption: two-column conference style (ACL/EMNLP-like), 10 pages + references/appendix extra. If one column, shrink each page's content ~40%.

Working title: **SnapRAG: Verified Reconstruction of Snapshot Memory for Long-Conversation Document QA**

Claims (from `SNAPRAG_PLAN.md`):
- **C1 (main)** Verified reconstruction: lossy snapshot → rebuilt text → every sentence checked against snapshot, document pages, cold archive.
- **C2 (small)** CIS snapshot trigger + turn-aware weight λ(t).
- **C3 (benchmark)** SnapRAG-LongConv: 100-turn document-QA conversations with memory probes.
- **C4 (small, unvalidated)** Intent- and memory-aware AB-RAG (AB-RAG itself is prior work).

---

## Page 1: Title, abstract, start of introduction

**Contains:** title, authors, abstract (150-200 words), keywords, first half of Introduction.

**Abstract skeleton (fill last):**
> Long conversations about documents outgrow the context window. Summaries and compressed memories save tokens but silently lose or distort facts, and the model then answers from the distorted memory with full confidence. We present SnapRAG, which keeps every turn in an immutable cold archive and a small lossy snapshot (facts, numbers, user statements, decisions, open questions, with doc/page anchors). When old information is needed, text is reconstructed from the snapshot and every sentence is verified against the snapshot, the anchored document pages and the archive; only verified sentences enter the prompt. A context-quality score (CIS) decides when to snapshot, a turn-aware weight λ(t) decides how much memory fits, and retrieval depth adapts to the question intent. We also release SnapRAG-LongConv, a 100-turn document-QA benchmark with 24 memory probes at four distances. On [document, LLM], SnapRAG reaches [X]% probe accuracy versus [Y]% for [baseline] at [Z]x fewer prompt tokens, and reduces unsupported memory claims from [A]% to [B]%.

**Keywords:** conversational memory, retrieval-augmented generation, hallucination, long-context, document QA.

---

## Page 2: Introduction (rest) and contributions

**Contains:** motivation, problem, gap, contributions, paper outline.

**Facts:**
- Motivating example: a fact stated at turn 5 (e.g. "my draft deadline is 15 November") is lost or distorted by turn 80.
- Failure modes seen in this project's own tests: decision buried by snapshot-level similarity; false premise endorsed; memory used without being checked.
- Existing fixes: bigger context (cost, lost-in-the-middle), summaries (lossy), memory stores (no check that recalled text is true).

**Skeleton:**
1. Paragraph: long document-QA chats are common (study, legal, support); context is finite.
2. Paragraph: compression is lossy and the loss is silent.
3. Paragraph: our idea in two sentences (*compress, reconstruct later, check against evidence, only then use*).
4. Contributions (bullets): C1, C2, C3, C4 with one line each. State C4 builds on AB-RAG.
5. One sentence: roadmap.

**Figure 1 (top of page 2 or 1):** the failure example, side by side: summary memory says wrong thing vs SnapRAG verified answer.

---

## Page 3: Related work

**Contains:** a table and 4 short paragraphs. Be fair: say what each does and the difference.

| Work | What it does | Difference from SnapRAG |
|---|---|---|
| LCM (Lossless Context Management) | immutable store + expand from summary | no verification of reconstructed text |
| MemCoRe | keyword/entity lossy compression | no verify step |
| ChronoMem, Cognis | versioned memory | no reconstruction check |
| Mem0, A-MEM | agent memory stores | no archive-backed check, no page anchors |
| LLMLingua, ICAE, Recursive Summarization | prompt compression | lossy, unchecked |
| LoCoMo, MTRAG | conversation benchmarks | personal chat / ~8 turns, no page citations |
| CRAG, Self-RAG, AB-RAG | confidence-gated retrieval | basis of C4; AB-RAG is prior work |

**Paragraphs:** (a) context compression, (b) conversational memory, (c) retrieval control and hallucination checking, (d) benchmarks.
**Before submitting:** literature search (Google Scholar, Semantic Scholar, arXiv) for "verified reconstruction", "memory hallucination", "context quality score", "intent-aware adaptive retrieval". Read LCM and AB-RAG in full. Have a second person check this table.

---

## Page 4: Problem setup and system overview

**Contains:** formal setting, notation, architecture figure.

**Facts:**
- Input: conversation of T turns (q_t, a_t) over document set D; token budget B for memory in the prompt.
- Goal: answer q_t using D and facts from earlier turns, with correct page citations, within B, without unsupported claims about the past.
- Stores: **cold archive** (append-only JSONL, immutable), **hot snapshots** (hash-chained, versioned, vector-indexed), **conversation state** (user statements/decisions/open questions, versioned).

**Figure 2:** architecture/flow (take from README): Query → Router (intent) → Retrieve ‖ Recall → Generate → Check → (extra batch) → Answer; after turn: CIS → snapshot / state update.

---

## Page 5: Method I: snapshots and the CIS trigger (C2)

**Contains:** snapshot content, CIS formula, trigger rule, λ(t).

**Facts:**
- Snapshot facets: facts, numbers, entities, user facts, decisions, open questions, asked topics; each with `doc/page` anchors; hash-chained.
- `CIS = 0.35·grounding + 0.25·headroom + 0.25·coherence − 0.15·contradiction`.
- Snapshot when `headroom < 0.20` or `CIS < 0.55`.
- `λ(t) = sigmoid(0.08·(t − 8))`: weight of memory versus fresh retrieval grows with turn number.
- Defaults are uncalibrated; ablations: `full_fixed_interval`, `full_fixed_lambda`.

**Skeleton:** Definition of each CIS term (what is measured and how), algorithm box "Snapshot trigger", equation for λ(t), one sentence why not a fixed interval.

---

## Page 6: Method II: verified reconstruction (C1, the core page)

**Contains:** the algorithm, the verification cascade, the drop rule. This is the most important page; give it the best figure.

**Facts:**
- Recall modes (ladder): `snapshot` → `evidence` → `verify` → `archive`.
- Reconstruction: LLM rebuilds text from snapshot items relevant to the query.
- Verification cascade per sentence: supported by snapshot? by anchored document pages? by archive turns? If none → **dropped**.
- Only verified sentences are injected into the prompt. Archive is the final authority because it is immutable.
- Cost: extra LLM calls; report them.

**Algorithm 1 (write as pseudocode):**
```
recall(query):
  items  = search snapshots(query)
  draft  = reconstruct(items)
  for s in sentences(draft):
      if supported(s, items) or supported(s, pages(items)) or supported(s, archive): keep s
      else: drop s
  return kept sentences within budget λ(t)·B
```
**Figure 3:** one example: reconstructed sentence kept (with source) and one dropped.

---

## Page 7: Method III: conversation state and AB-RAG (C4)

**Contains:** two sub-sections.

**Conversation state (half page):**
- Candidates (user statements, decisions, open questions) are extracted, then each is **verified against the user's own messages**; the LLM may only point at superseded items; versions kept in `state.jsonl`.
- Hand-off document built on request; injected when a session resumes after a gap.

**AB-RAG (half page):**
- `confidence = 0.50·S1 + 0.25·S2 + 0.25·S3`
  - S1: checker's judged certainty (a **proxy**: gateway gives no token probabilities)
  - S2: cosine of the answer to the closest evidence (documents or verified memory)
  - S3: spread (std/3) of reranker scores
- Answer not supported → confidence capped at 0.79. Stop when ≥ 0.80.
- Added here: intent sets first batch and budget (e.g. DEFINITION 3 passages + 1 extra batch, SUMMARY 8 + 3); verified memory counts as evidence; CONVERSATION questions skip the documents; next batch = same ranking read deeper (free), then fresh wide search (HyDE, 2× candidates).
- State clearly: AB-RAG is prior work (arXiv 2606.29090); weights and threshold are uncalibrated.

**Table 1:** intent → first batch, extra-batch budget (from `query_router.py`).

---

## Page 8: Benchmark SnapRAG-LongConv (C3)

**Contains:** construction, probes, scoring, variants.

**Facts:**
- Scripted conversation of 100 turns (minimum 80) about the Chapin torts book (PDF page = printed page + 20); document-QA turns interleaved with 12 plants (user facts) and 2 corrections.
- 24 probes: recall, supersede, reference, absent, trap; distance buckets near / mid / far / very_far.
- Scoring: regex per probe (deterministic, no LLM judge); bootstrap confidence intervals; paired differences between variants; `--repeats 3`.
- Dry-run oracle checks the scoring itself.

**Table 2: variants**

| Group | Variants |
|---|---|
| Baselines | `no_memory`, `full_history`, `summary_buffer` |
| Ladder (C1) | `snapshot`, `evidence`, `verify`, `archive` |
| Others | `state_only`, `full` |
| Ablations | `full_fixed_lambda`, `full_no_exemplars`, `full_fixed_interval`, `full_fixed_k` |

**Setup paragraph (fill):** LLM `[model/gateway]`, embedder `[name]`, reranker `[name]`, Qdrant, temperature, seeds, date.

---

## Page 9: Results

**Contains:** main table, figure, ablations. Report numbers with intervals.

**Table 3: main results** (one row per variant)

| Variant | Probe acc. [CI] | near | mid | far | very far | Memory loss (near − very far) | Prompt tokens/turn | LLM calls/turn |
|---|---|---|---|---|---|---|---|---|
| no_memory | [ ] | | | | | | | |
| full_history | [ ] | | | | | | | |
| summary_buffer | [ ] | | | | | | | |
| full | [ ] | | | | | | | |

**Table 4: ablations** as paired differences with CI: ladder (C1), `full_fixed_interval`/`full_fixed_lambda` (C2), `full_fixed_k` (C4: accuracy, page-hit, passages, calls), `full_no_exemplars`.

**Figure 4:** accuracy versus distance bucket, one line per variant (main figure).
**Figure 5:** ladder bars: unsupported memory claims and fact recall (`reconstruction_eval`).

**Analysis paragraphs:** what helped, what did not, failure cases (list real misses from the result JSON: which probe, answer, why). Latency note: gateway latency grew with turn count in our runs, report p50/p95.
Current state (unfinished run, not for the paper): `full` 16/21 probes, misses on a trap probe and four recalls.

---

## Page 10: Limitations, conclusion

**Contains:** honest limits, conclusion, then references (not counted) and appendix.

**Limitations (write all that apply):** one document and one domain; one LLM backbone; LLM-based verification can itself err; S1 is a proxy; AB-RAG weights/threshold uncalibrated; extra LLM calls and latency; scripted (synthetic) conversations; LCM, Mem0, A-MEM, LLMLingua not run as baselines unless done with their own code; novelty claims rest on a limited search.

**Conclusion skeleton:** one sentence problem, one sentence method, one sentence best measured result, one sentence future work (second document, second backbone, calibrate AB-RAG, human study).

**Claim carefully:** "near full-context accuracy at fewer tokens, with fewer unsupported memory claims". Do **not** write "most accurate" or "first".

---

## Appendix (beyond 10 pages)
A. Prompts (snapshot, reconstruction, verification, generation). B. Full probe list and regexes. C. Hyperparameters and all settings. D. Per-probe results of every variant. E. Cost table. F. Reproduction commands:
```
python -m benchmarks.longconv.run --variants main --turns 100 --repeats 3 --pause 2
python -m benchmarks.reconstruction_eval
```

## Checklist before writing numbers
1. Run `main` variants with `--repeats 3` (at least `full`, `no_memory`, `full_history`, `summary_buffer`, `full_fixed_k`).
2. Check misses in the result JSON; fix real bugs first, then re-run (do not tune on the test probes without saying so).
3. Run the literature check and fill the related-work table.
4. Calibrate AB-RAG on a separate held-out script, not on the reported one.
5. Fill every `[ ]`; delete any claim the data does not support.

## Page budget summary
| Page | Section |
|---|---|
| 1 | Title, abstract, intro start |
| 2 | Intro, contributions |
| 3 | Related work |
| 4 | Problem setup, architecture |
| 5 | Snapshots, CIS, λ(t) |
| 6 | Verified reconstruction |
| 7 | Conversation state, AB-RAG |
| 8 | Benchmark |
| 9 | Results |
| 10 | Limitations, conclusion |
