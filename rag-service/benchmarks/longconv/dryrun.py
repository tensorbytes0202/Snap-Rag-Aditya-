"""A stand-in for the RAG system, to check the benchmark itself without any
service, model or API key (`run --dry-run`).

It answers a probe correctly with the probe's gold answer if the fact it
asks for was stated within the last `remembers` turns, and otherwise like a
system that forgot. That gives the benchmark a known memory curve to measure:
a run that does not show it points at a bug in the harness, not in SnapRAG.
"""


class DryRunSystem:

    def __init__(self, script: list[dict], remembers: int = 40):
        self.by_question = {item["text"]: item for item in script}
        self.remembers = remembers

    def answer(self, question: str, turn: int) -> dict:

        item = self.by_question.get(question)

        if item is None or item["kind"] != "probe":
            return self._response("Insufficient information found." if item and item["kind"] != "doc" else "An answer about the document (PDF p. 100).")

        if item["probe_kind"] in ("absent", "trap"):
            return self._response(item["gold"])

        remembered = turn - max(item["source_turns"]) <= self.remembers

        return self._response(item["gold"] if remembered else "Insufficient information found.")

    @staticmethod
    def _response(answer: str) -> dict:

        return {
            "answer": answer,
            "citations": [{"source_id": 1, "document": "dry-run.pdf", "page": 100}],
            "verification": {"supported": True, "confidence": 0.9},
            "confidence": 0.9,
            "retrieved_memory": None,
        }
