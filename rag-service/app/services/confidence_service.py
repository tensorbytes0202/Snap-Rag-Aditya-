import math


def calculate_confidence(rerank_scores):

    if not rerank_scores:
        return 0.0

    best = max(rerank_scores)

    confidence = 1 / (1 + math.exp(-(best + 1.5)))

    confidence = max(0.05, min(confidence, 0.99))

    return round(confidence, 2)