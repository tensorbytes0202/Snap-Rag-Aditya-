# Self-review (a stand-in, not an outside read)

Written by the person who built the system, as a hostile reviewer would write it. It does not replace a mock review by someone else.

## What a reviewer will attack first

1. **One model family, free tier, small LLMs.** Everything runs on Gemini flash-lite models. The re-implemented baselines (Self-RAG, CRAG, ...) are
   prompted versions on a small model; a result against them says little about the published systems. *Mitigation in the paper:* labelled as
   re-implementations everywhere; no claim of beating the original methods.
2. **Few questions and probes.** 75 questions, 24 probes per conversation, one repeat per variant. Most paired differences have intervals that include zero.
   *Mitigation:* exact intervals and Holm adjustment are reported, differences that are not clearly better or worse are called noise.
3. **The verifier is an LLM.** C1 rests on an LLM judging sentences. *Mitigation:* the ladder uses a different, stronger judge model than the reconstruction
   model, and the verifier's own false-accept and false-reject rates are measured on a hand-labelled sample. The labeller is one person (not independent),
   which must be said.
4. **Scripted conversations.** Users do not speak like the script, and "plants" are explicit ("note this for later"). Probes check recall, not whether the
   system notices an unmarked fact. *Mitigation:* stated as a limitation; no claim about natural conversations.
5. **Strict scoring.** Every fact must be in the answer, so near-misses count as wrong, and the patterns were written by the authors.
6. **AB-RAG differs from the paper.** S1 is a judged confidence, not token probabilities or self-consistency. The weights are ours, calibrated on held-out
   questions only afterwards. The original is only known from its abstract until the full text is read.
7. **Novelty.** The searches were short and used abstracts. Context Codec (2605.17304) is close; if its full text already verifies reconstructed sentences against
   an archive, C1 is not new. C2 (CIS, lambda) is a set of hand-set coefficients with no fitted justification.
8. **Cost.** The checks add LLM calls and latency (see the cost tables). If a plain hybrid + cross-encoder pipeline is as accurate for a third of the calls,
   the adaptive part is not worth it on this data; the results say whether that is so.
9. **Free-tier artefacts.** Rate limits and quota failures shaped the runs: failed turns are excluded from the probe scores, and wall-clock times are not
   comparable between methods (active seconds are reported separately).
10. **Licence.** *A Suitable Boy* is under copyright; only questions, answers and page numbers are released.

## What would most strengthen the paper (not done yet)

- A larger or different LLM, repeats of every variant, and a second conversation script (a different seed) with a different document.
- A second labeller for the verifier sample.
- Running Mem0, A-MEM, LCM and LLMLingua with their own code.
- A short human study: do people prefer the answers of the verified memory?
- Reading the full texts of AB-RAG and Context Codec, and a Scholar / Semantic Scholar search.
