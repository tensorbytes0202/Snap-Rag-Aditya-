"""The RAG architectures compared by the document-QA benchmark.

All of them use the same index (Qdrant dense search and BM25 over the same
chunks), the same embedder, cross-encoder and answer prompt, and the same LLM,
so a difference in the results comes from the architecture alone.

Standard pipelines:
    closed_book          no retrieval: the LLM answers from what it knows of the book
    dense_k5             top-5 pages by embedding similarity (naive RAG)
    bm25_k5              top-5 pages by keyword score
    hybrid_k5            dense + BM25 fused with RRF, top 5
    hybrid_rerank_k5     hybrid, then the cross-encoder re-ranks 20 candidates, top 5
    hybrid_rerank_k10    the same with 10 pages (a stronger fixed-k baseline)

Published methods, re-implemented here in the same harness from their papers'
descriptions (simplified; they are NOT the authors' trained models or code, and
they run on the same small LLM as everything else):
    hyde                 HyDE (Gao et al. 2022): retrieve with an LLM-written hypothetical answer
    crag                 CRAG (Yan et al. 2024): grade the retrieved pages, re-search when none is relevant
    self_rag             Self-RAG (Asai et al. 2023), prompted: one answer per page, a critic picks the best
    ircot                IRCoT (Trivedi et al. 2022): reasoning steps interleaved with retrieval
    adaptive_rag         Adaptive-RAG (Jeong et al. 2024): route by question complexity
    abrag                AB-RAG (arXiv 2606.29090): answer, score the confidence, read more while it is low
                         (one fixed first batch and budget for every question)

SnapRAG:
    snaprag_fixed        its own retrieval (planner, intent-sized batch, HyDE) plus
                         verification, but no extra batches: fixed-k, AB-RAG off
    snaprag              the full pipeline: intent-sized batches, AB-RAG on top

Each architecture returns {"answer", "chunks", "context"} (the pages the LLM read)
and, for the AB-RAG based ones, "confidence" and "supported".
"""

from app.core.config import settings
from app.services.abrag import run_batches
from app.services.answering import make_attempt, present_answer
from app.services.bm_25_service import bm25_search
from app.services.generation_service import generate_answer
from app.services.hyde_service import generate_hyde
from app.services.llm_provider import chat
from app.services.metadata_parser import extract_metadata
from app.services.query_planner import fallback_subqueries, guess_query_type, plan_query
from app.services.query_router import get_retrieval_config
from app.services.query_runner import Retrieval, run_retrieval
from app.services.reranker_services import rerank_chunks
from app.services.retrieval_pipeline import retrieve_chunks
from app.services.rrf_service import rrf_fusion
from app.utils.json_utils import extract_json


POOL = 20     # candidates each index returns before fusion / re-ranking

CLOSED_BOOK_PROMPT = (
    "Answer the question about Vikram Seth's novel A Suitable Boy from your own knowledge only. "
    "If you do not know or are not sure, say that you do not know.\n\nQuestion: {question}"
)

GRADE_PROMPT = """You grade retrieved passages for a question about a novel.

Question: {question}

{passages}

For each passage say "relevant" (it holds information needed to answer), "ambiguous" (related, but not clearly
enough) or "irrelevant". Return only JSON: {{"grades": ["relevant", "irrelevant"]}} with one entry per passage, in order."""

REWRITE_PROMPT = "Rewrite the question as a short keyword search query for finding the answer in a novel. Return only the query.\n\nQuestion: {question}"

CRITIC_PROMPT = """Question about a novel: {question}

Each candidate answer below was written from one passage only.

{candidates}

For every candidate rate from 1 to 5: how relevant the passage is to the question, how well the passage supports the answer,
and how useful the answer is. Return only JSON: {{"scores": [[relevance, support, usefulness], ...]}} with one entry per candidate."""

ROUTE_PROMPT = """Classify the question about a novel by the retrieval it needs.
A: simple, answerable without looking anything up
B: needs facts from one place in the book (single-step retrieval)
C: needs facts from several places or several reasoning steps (multi-step retrieval)

Question: {question}

Return only the letter."""

STEP_PROMPT = """Answer the question about a novel from the passages, reasoning step by step.
Write ONLY the next reasoning sentence. If the passages already let you answer fully, write ANSWER: and the final answer.

Question: {question}

Passages:
{context}

Reasoning so far: {thoughts}"""

# AB-RAG as published: the same first batch and budget for every question (no intent routing).
PLAIN_CONFIG = {"dense_top_k": 20, "bm25_top_k": 10, "use_hyde": False, "passages": 5, "batches": 3}


def _ask(prompt: str) -> str:

    return chat([{"role": "user", "content": prompt}])["message"]["content"]


def _unique(chunks):
    """One chunk per page passage, in order."""

    seen, kept = set(), []

    for chunk in chunks:
        key = (chunk.get("page"), chunk.get("parent_id"))
        if key not in seen:
            seen.add(key)
            kept.append(chunk)

    return kept


def _dense(question, document, k):

    return retrieve_chunks([question], {"document_name": document, "page": None}, {"dense_top_k": k})


def _bm25(question, document, k):

    return [{**chunk, "bm25_score": float(score)} for chunk, score in bm25_search(question, document, k)]


def _hybrid(question, document):

    return rrf_fusion(_dense(question, document, POOL), _bm25(question, document, POOL))


def _rerank(question, document):

    return rerank_chunks(question, _hybrid(question, document)[:POOL])


def _context(question, document, chunks):
    """The pages as the LLM reads them, and the chunks actually used."""

    found = Retrieval([question], "FACTUAL", {}, ranked_chunks=list(chunks), limit=len(chunks))

    for chunk in found.top_chunks:
        chunk.setdefault("document", document)

    return found.top_context, found.top_chunks


def _read(question, document, chunks, k, offline=False):
    """The top `k` pages, and the answer the LLM gives from them."""

    context, used = _context(question, document, list(chunks)[:k])
    answer = None if offline else generate_answer(question, context, "FACTUAL")

    return {"answer": answer, "chunks": used, "context": context}


def _fixed(ranking, k):
    return lambda question, document, offline=False: _read(question, document, ranking(question, document), k, offline)


def _closed_book(question, document, offline=False):

    if offline:
        return {"answer": None, "chunks": [], "context": ""}

    return {"answer": _ask(CLOSED_BOOK_PROMPT.format(question=question)), "chunks": [], "context": ""}


# ---------------- published methods ----------------

def _hyde(question, document, offline=False):

    return _read(question, document, _dense(generate_hyde(question), document, 5), 5)


def _passage_text(chunk, limit=700):

    return " ".join((chunk.get("parent_text") or chunk["text"]).split())[:limit]


def _crag(question, document, offline=False):
    """Correct: keep the relevant pages. Otherwise the question is rewritten and searched again
    (the paper's web search has no counterpart: the book is the only corpus)."""

    candidates = _unique(_rerank(question, document))[:5]
    listing = "\n\n".join(f"Passage {i}: {_passage_text(c)}" for i, c in enumerate(candidates, 1))
    data = extract_json(_ask(GRADE_PROMPT.format(question=question, passages=listing)))
    grades = data.get("grades") if isinstance(data, dict) else None
    grades = [str(g).lower() for g in grades] if isinstance(grades, list) and len(grades) == len(candidates) else ["relevant"] * len(candidates)

    relevant = [c for c, g in zip(candidates, grades) if g == "relevant"]

    if relevant:
        return _read(question, document, relevant, 5)

    ambiguous = [c for c, g in zip(candidates, grades) if g == "ambiguous"]
    query = _ask(REWRITE_PROMPT.format(question=question)).strip()

    return _read(question, document, _unique(ambiguous + _rerank(query, document)), 5)


def _self_rag(question, document, offline=False):
    """Prompted Self-RAG: each of the top 3 pages gives its own answer, a critic scores
    relevance / support / usefulness, and the best-scored answer is returned."""

    pages = _unique(_rerank(question, document))[:3]
    drafts = [_read(question, document, [page], 1) for page in pages]
    listing = "\n\n".join(f"Candidate {i}\nPassage: {_passage_text(d['chunks'][0], 500)}\nAnswer: {' '.join(d['answer'].split())[:500]}"
                          for i, d in enumerate(drafts, 1))
    data = extract_json(_ask(CRITIC_PROMPT.format(question=question, candidates=listing)))
    scores = data.get("scores") if isinstance(data, dict) else None

    try:
        totals = [sum(float(x) for x in row) for row in scores]
        best = totals.index(max(totals)) if len(totals) == len(drafts) else 0
    except (TypeError, ValueError):
        best = 0

    return {"answer": drafts[best]["answer"], "chunks": [d["chunks"][0] for d in drafts],
            "context": "\n\n".join(d["context"] for d in drafts)}


def _ircot(question, document, offline=False, steps=3):
    """Interleaves retrieval with reasoning: each reasoning sentence becomes the next search query."""

    evidence = _unique(_rerank(question, document))[:3]
    thoughts = []

    for _ in range(steps):

        context, _ = _context(question, document, evidence)
        reply = _ask(STEP_PROMPT.format(question=question, context=context, thoughts=" ".join(thoughts) or "(none)"))

        if "ANSWER:" in reply.upper():
            break

        thoughts.append(" ".join(reply.split())[:300])
        evidence = _unique(evidence + _unique(_rerank(thoughts[-1], document))[:3])[:9]

    return _read(question, document, evidence, len(evidence))


def _adaptive_rag(question, document, offline=False):
    """Routes by question complexity: no retrieval, single-step retrieval, or multi-step (IRCoT)."""

    reply = _ask(ROUTE_PROMPT.format(question=question)).strip().upper()
    route = next((letter for letter in reply if letter in "ABC"), "B")

    if route == "A":
        out = _closed_book(question, document)
    elif route == "C":
        out = _ircot(question, document)
    else:
        out = _read(question, document, _rerank(question, document), 5)

    return {**out, "route": route}


# ---------------- AB-RAG based ----------------

def _adaptive(question, document, offline=False, plan=None):
    """The query pipeline of query_service.process_query without memory, follow-ups
    and the answer cache (a single-turn question has none of them). With a `plan`
    the intent routing is replaced by that fixed plan."""

    found = run_retrieval(question, document, plan=plan)

    if offline:
        return {"answer": None, "chunks": found.top_chunks, "context": found.top_context}

    if not found.ranked_chunks:
        return {"answer": "No relevant information found in the uploaded documents.", "chunks": [], "context": ""}

    first = make_attempt(question, found, None, None, "initial")
    attempts = [first]

    def deeper_attempt(deeper, strategy):
        attempts.append(make_attempt(question, deeper, None, None, strategy))
        return attempts[-1]

    best, _ = run_batches(
        question,
        first,
        lambda **overrides: run_retrieval(question, document, **overrides),
        deeper_attempt,
    )

    return {
        "answer": present_answer(best),
        "chunks": best.found.top_chunks,
        "context": best.found.top_context,
        "confidence": best.confidence,
        "supported": best.supported,
        # Every attempt with its confidence signals and answer, so a threshold and the
        # weights can be replayed offline (see calibrate.py).
        "attempts": [{"strategy": a.strategy, "confidence": a.confidence, "signals": a.signals, "supported": a.supported,
                      "passages": len(a.found.top_chunks), "answer": a.answer} for a in attempts],
    }


def _plan(question, planner: bool) -> dict:
    """A query plan without HyDE: from the LLM planner, or the heuristic one (no LLM)."""

    plan = plan_query(question) if planner else {"query_type": guess_query_type(question), "subqueries": fallback_subqueries(question)}
    metadata = extract_metadata(question) or {}
    metadata["document_name"] = None
    metadata.setdefault("page", None)

    return {"query_type": plan["query_type"], "subqueries": plan["subqueries"], "hyde": None, "metadata": metadata,
            "config": get_retrieval_config(plan["query_type"])}


def _retrieval_only(plan_for):
    """SnapRAG's retrieval alone (nothing is answered): to see what its planner and HyDE do to the pages read."""

    def run(question, document, offline=False):
        found = run_retrieval(question, document, plan=plan_for(question))
        return {"answer": None, "chunks": found.top_chunks, "context": found.top_context}

    return run


def _no_planner(question, document, offline=False):

    return _adaptive(question, document, offline, plan=_plan(question, planner=False))


def _abrag(question, document, offline=False):

    plan = {"query_type": "FACTUAL", "subqueries": [question], "hyde": None,
            "metadata": {"page": None, "document_name": None}, "config": PLAIN_CONFIG}

    return _adaptive(question, document, offline, plan=plan)


# name -> (description, function, settings overrides, source)
ARCHITECTURES = {
    "closed_book": ("no retrieval", _closed_book, {}, "baseline"),
    "dense_k5": ("dense top-5", _fixed(lambda q, d: _dense(q, d, 5), 5), {}, "naive RAG"),
    "bm25_k5": ("BM25 top-5", _fixed(lambda q, d: _bm25(q, d, 5), 5), {}, "baseline"),
    "hybrid_k5": ("dense + BM25 (RRF) top-5", _fixed(_hybrid, 5), {}, "standard hybrid RAG"),
    "hybrid_rerank_k5": ("hybrid + cross-encoder top-5", _fixed(_rerank, 5), {}, "standard advanced RAG"),
    "hybrid_rerank_k10": ("hybrid + cross-encoder top-10", _fixed(_rerank, 10), {}, "standard advanced RAG, more pages"),
    "hyde": ("retrieve with a hypothetical answer", _hyde, {}, "HyDE, Gao et al. 2022 (arXiv 2212.10496)"),
    "crag": ("grade the pages, re-search if none is relevant", _crag, {}, "CRAG, Yan et al. 2024 (arXiv 2401.15884); no web search, no strip refinement"),
    "self_rag": ("one answer per page, a critic picks one", _self_rag, {}, "Self-RAG, Asai et al. 2023 (arXiv 2310.11511); prompted, not the fine-tuned model"),
    "ircot": ("reasoning steps interleaved with retrieval", _ircot, {}, "IRCoT, Trivedi et al. 2022 (arXiv 2212.10509); 3 steps, no few-shot demos"),
    "adaptive_rag": ("route by question complexity", _adaptive_rag, {}, "Adaptive-RAG, Jeong et al. 2024 (arXiv 2403.14403); LLM prompt instead of the trained classifier"),
    "abrag": ("answer, score confidence, read more while it is low", _abrag, {}, "AB-RAG, arXiv 2606.29090 (known from its summary); same confidence signals, one fixed first batch and budget"),
    "snaprag_fixed_k": ("SnapRAG retrieval + confidence check, no extra batches (fixed-k)", _adaptive, {"MAX_EXTRA_BATCHES": 0}, "this work"),
    "snaprag": ("SnapRAG: intent-sized batches + AB-RAG", _adaptive, {}, "this work"),
    # Diagnostics, not competitors:
    "snaprag_fixed_k_noplan": ("fixed-k SnapRAG with the heuristic plan (no LLM planner, no HyDE)", _no_planner, {"MAX_EXTRA_BATCHES": 0}, "ablation"),
    "snaprag_retrieval": ("SnapRAG retrieval only: LLM planner + HyDE, nothing answered", _retrieval_only(lambda q: None), {}, "ablation"),
    "snaprag_retrieval_planner": ("SnapRAG retrieval only: LLM planner, no HyDE", _retrieval_only(lambda q: _plan(q, True)), {}, "ablation"),
    "snaprag_calib": ("SnapRAG that never stops early: every attempt is kept for calibration", _adaptive, {"STOP_CONFIDENCE": 1.01}, "calibration"),
}

# Not part of the comparison tables: they feed the planner ablation and the calibration.
DIAGNOSTIC = ("snaprag_fixed_k_noplan", "snaprag_retrieval", "snaprag_retrieval_planner", "snaprag_calib")

# The architectures whose retrieval can be measured without any LLM call
# (SnapRAG falls back to its offline planner there).
RETRIEVAL_ONLY = ["dense_k5", "bm25_k5", "hybrid_k5", "hybrid_rerank_k5", "hybrid_rerank_k10", "snaprag_fixed_k"]


def apply_settings(overrides: dict) -> dict:
    """Sets `overrides` on the shared settings; returns the previous values."""

    previous = {key: getattr(settings, key) for key in overrides}

    for key, value in overrides.items():
        setattr(settings, key, value)

    return previous
