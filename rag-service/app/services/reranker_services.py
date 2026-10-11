from langsmith import traceable
from app.services import cross_encoder_service
from app.api.dependencies.reranker import get_cross_encoder


@traceable(name="rerank_chunks", run_type="chain")
def rerank_chunks(
    question,
    chunks
):

    if not chunks:
        return []

    # ==========================================
    # Prepare Cross Encoder Input
    # ==========================================

    pairs = [

        (question, chunk["text"])

        for chunk in chunks

    ]
    cross_encoder = get_cross_encoder()
    scores = cross_encoder.predict(
        pairs
    )

    ranked = []

    # ==========================================
    # Attach Score
    # ==========================================

    for chunk, score in zip(chunks, scores):

        chunk = dict(chunk)

        chunk["rerank_score"] = float(score)

        ranked.append(chunk)

    # ==========================================
    # Sort
    # ==========================================

    ranked.sort(

        key=lambda x: x["rerank_score"],

        reverse=True

    )

    return ranked