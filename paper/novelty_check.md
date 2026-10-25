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
