def get_retrieval_config(query_type):

    configs = {

        "DEFINITION": {

            "dense_top_k": 5,
            "bm25_top_k": 3,
            "compression_top_k": 3,
            "window_size": 1,
            "use_hyde": False

        },

        "EXPLANATION": {

            "dense_top_k": 8,
            "bm25_top_k": 5,
            "compression_top_k": 5,
            "window_size": 2,
            "use_hyde": True

        },

        "COMPARISON": {

            "dense_top_k": 10,
            "bm25_top_k": 8,
            "compression_top_k": 6,
            "window_size": 2,
            "use_hyde": True

        },

        "LIST": {

            "dense_top_k": 12,
            "bm25_top_k": 10,
            "compression_top_k": 8,
            "window_size": 2,
            "use_hyde": True

        },

        "PROCEDURE": {

            "dense_top_k": 10,
            "bm25_top_k": 8,
            "compression_top_k": 6,
            "window_size": 2,
            "use_hyde": True

        },

        "SUMMARY": {

            "dense_top_k": 20,
            "bm25_top_k": 15,
            "compression_top_k": 10,
            "window_size": 3,
            "use_hyde": True

        },

        "CODE": {

            "dense_top_k": 10,
            "bm25_top_k": 8,
            "compression_top_k": 6,
            "window_size": 2,
            "use_hyde": False

        },

        "FACTUAL": {

            "dense_top_k": 3,
            "bm25_top_k": 3,
            "compression_top_k": 2,
            "window_size": 0,
            "use_hyde": False

        }

    }

    return configs.get(
        query_type,
        configs["EXPLANATION"]
    )