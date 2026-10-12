def get_retrieval_config(query_type):

    configs = {

        "DEFINITION": {

            "dense_top_k": 5,
            "bm25_top_k": 3,
            "use_hyde": False

        },

        "EXPLANATION": {

            "dense_top_k": 8,
            "bm25_top_k": 5,
            "use_hyde": True

        },

        "COMPARISON": {

            "dense_top_k": 10,
            "bm25_top_k": 8,
            "use_hyde": True

        },

        "LIST": {

            "dense_top_k": 12,
            "bm25_top_k": 10,
            "use_hyde": True

        },

        "PROCEDURE": {

            "dense_top_k": 10,
            "bm25_top_k": 8,
            "use_hyde": True

        },

        "SUMMARY": {

            "dense_top_k": 20,
            "bm25_top_k": 15,
            "use_hyde": True

        },

        "CODE": {

            "dense_top_k": 10,
            "bm25_top_k": 8,
            "use_hyde": False

        },

        "FACTUAL": {

            "dense_top_k": 3,
            "bm25_top_k": 3,
            "use_hyde": False

        }

    }

    return configs.get(
        query_type,
        configs["EXPLANATION"]
    )