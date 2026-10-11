from app.api.dependencies.embedding import get_embedding_model

from app.cache.embedding_cache import (
    get_embedding,
    set_embedding
)


# ==========================================================
# Document Embeddings
# ==========================================================

def generate_embeddings(chunks):

    if not chunks:
        return []

    model = get_embedding_model()

    embeddings = model.encode(
        chunks,
        normalize_embeddings=True
    )

    return embeddings.tolist()


# ==========================================================
# Query Embedding (with Redis Cache)
# ==========================================================

def generate_query_embedding(query: str):

    # Check Redis Cache
    cached = get_embedding(query)

    if cached is not None:
        return cached

    model = get_embedding_model()

    embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Save to Redis
    set_embedding(
        query,
        embedding
    )

    return embedding