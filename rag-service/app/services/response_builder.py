# ==========================================================
# Success Response
# ==========================================================

def build_success_response(
    question,
    rewritten_question,
    answer,
    followups,
    verification,
    confidence,
    results,
    bm25_chunks,
    hybrid_chunks,
    sources,
    citations,
    memory=None,
    attempts=None,
    signals=None,
):

    return {
        "question": question,
        "rewritten_question": rewritten_question,
        "answer": answer,
        "followup_questions": followups,
        "verification": verification,
        "confidence": confidence,
        "retrieved_dense": len(results),
        "retrieved_bm25": len(bm25_chunks),
        "retrieved_hybrid": len(hybrid_chunks),
        "sources": sources[:5],
        "citations": citations,
        # Verified snapshot memory used for this answer (None outside a chat).
        "retrieved_memory": memory,
        # Every attempt (strategy, confidence, passages and pages read), and the
        # confidence signals of the returned answer.
        "attempts": attempts or [],
        "signals": signals or {},
    }


# ==========================================================
# Failure Response
# ==========================================================

def build_failure_response(question, rewritten_question):

    return {
        "question": question,
        "rewritten_question": rewritten_question,
        "answer": "No relevant information found in the uploaded documents.",
        "verification": "FAILED",
        "confidence": 0,
        "retrieved_dense": 0,
        "retrieved_bm25": 0,
        "retrieved_hybrid": 0,
        "sources": [],
        "citations": [],
    }
