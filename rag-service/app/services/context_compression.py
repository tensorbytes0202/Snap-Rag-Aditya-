from app.api.dependencies.reranker import get_cross_encoder


# ============================================
# Context Compression
# ============================================

def compress_context(
    question,
    chunks,
    top_k=8
):
    """
    Compress expanded context using
    Cross Encoder relevance scoring.
    """

    if not chunks:
        return []

    cross_encoder = get_cross_encoder()

    pairs = [
        (question, chunk)
        for chunk in chunks
    ]

    scores = cross_encoder.predict(pairs)

    ranked = list(zip(chunks, scores))

    ranked.sort(
        key=lambda x: x[1],
        reverse=True
    )

    compressed = [
        chunk
        for chunk, _ in ranked[:top_k]
    ]

    return compressed