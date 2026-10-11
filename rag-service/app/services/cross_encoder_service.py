from app.api.dependencies.reranker import get_cross_encoder


def rerank(query, documents):

    if not documents:
        return []

    cross_encoder = get_cross_encoder()

    sentence_pairs = [
        (query, doc["text"])
        for doc in documents
    ]

    scores = cross_encoder.predict(sentence_pairs)

    for doc, score in zip(documents, scores):
        doc["rerank_score"] = float(score)

    documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return documents