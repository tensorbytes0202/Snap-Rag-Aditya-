from langsmith import traceable
@traceable(name="rrf_fusion", run_type="chain")
def rrf_fusion(
    dense_chunks,
    bm25_chunks,
    k=60
):

    scores = {}
    objects = {}

    # =====================================================
    # Dense Retrieval
    # =====================================================

    for rank, chunk in enumerate(dense_chunks):

        key = chunk["text"]

        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        objects[key] = dict(chunk)

    # =====================================================
    # BM25 Retrieval
    # =====================================================

    for rank, chunk in enumerate(bm25_chunks):

        key = chunk["text"]

        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)

        if key in objects:

            # Merge metadata
            objects[key].update(chunk)

        else:

            objects[key] = dict(chunk)

    # =====================================================
    # Add RRF Score
    # =====================================================

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    fused = []

    for text, score in ranked:

        chunk = objects[text]

        chunk["rrf_score"] = score

        fused.append(chunk)

    return fused