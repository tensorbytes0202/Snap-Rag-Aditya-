"""Retrieval settings per query type (the intent).

    dense_top_k, bm25_top_k  how many candidates each index returns
    use_hyde                 add a HyDE passage to the dense search
    passages                 the first batch: passages the LLM reads for a one-topic question
    batches                  the budget: extra batches of passages the confidence check may
                             ask for (see abrag); MAX_EXTRA_BATCHES caps it

A definition needs little evidence and a comparison or summary a lot, so the first
batch and the budget follow the intent instead of one fixed k for every question.
"""

CONFIGS = {
    "DEFINITION":   {"dense_top_k": 5,  "bm25_top_k": 3,  "use_hyde": False, "passages": 3, "batches": 1},
    "FACTUAL":      {"dense_top_k": 3,  "bm25_top_k": 3,  "use_hyde": False, "passages": 3, "batches": 1},
    "CODE":         {"dense_top_k": 10, "bm25_top_k": 8,  "use_hyde": False, "passages": 4, "batches": 1},
    "EXPLANATION":  {"dense_top_k": 8,  "bm25_top_k": 5,  "use_hyde": True,  "passages": 5, "batches": 2},
    "PROCEDURE":    {"dense_top_k": 10, "bm25_top_k": 8,  "use_hyde": True,  "passages": 5, "batches": 2},
    "COMPARISON":   {"dense_top_k": 10, "bm25_top_k": 8,  "use_hyde": True,  "passages": 6, "batches": 3},
    "LIST":         {"dense_top_k": 12, "bm25_top_k": 10, "use_hyde": True,  "passages": 6, "batches": 3},
    "SUMMARY":      {"dense_top_k": 20, "bm25_top_k": 15, "use_hyde": True,  "passages": 8, "batches": 3},
    # About the chat itself: the documents rarely help, so search lightly and never search again.
    "CONVERSATION": {"dense_top_k": 5,  "bm25_top_k": 3,  "use_hyde": False, "passages": 3, "batches": 0},
}


def get_retrieval_config(query_type: str) -> dict:

    return CONFIGS.get(query_type, CONFIGS["EXPLANATION"])
