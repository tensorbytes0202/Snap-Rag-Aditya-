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

    dense_chunks,
    expanded_chunks,
    compressed_chunks,

    results,
    bm25_chunks,
    hybrid_chunks,

    sources,
    citations,
    context_chunks

):

    return {

        "question": question,

        "rewritten_question": rewritten_question,

        "answer": answer,

        "followup_questions": followups,

        "verification": verification,

        "confidence": confidence,

        "retrieved_parent": len(dense_chunks),

        "retrieved_expanded": len(expanded_chunks),

        "retrieved_compressed": len(compressed_chunks),

        "retrieved_dense": len(results),

        "retrieved_bm25": len(bm25_chunks),

        "retrieved_hybrid": len(hybrid_chunks),

        "sources": sources[:5],

        "citations": citations,

        "context_chunks": context_chunks

    }


# ==========================================================
# Failure Response
# ==========================================================

def build_failure_response(

    question,
    rewritten_question

):

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

        "context_chunks": []

    }