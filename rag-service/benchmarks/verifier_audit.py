"""Error rate of the sentence verifier, measured against hand labels.

The verified reconstruction keeps a sentence only when the verifier finds it supported by the snapshot, the
anchored document pages or the archive. This measures how often the verifier is wrong:

    false accept   a kept sentence that the sources do not support (an unsupported claim reaches the prompt)
    false reject   a dropped sentence that the sources do support (information is lost)

Step 1 samples kept and dropped sentences from a ladder result and writes a sheet with each sentence next to
its sources. Step 2 reads the labels (true = the sources support the sentence) and reports the error rates
with exact intervals.

    python -m benchmarks.verifier_audit sample benchmarks/results/ladder_full.json --per-side 30 --out audit_sheet.json
    python -m benchmarks.verifier_audit score audit_sheet.json
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.docqa.stats import exact_interval

VERIFIED_MODES = ("verify", "archive")       # the modes that check sentences


def sample(report: dict, per_side: int, seed: int = 0) -> list[dict]:
    """Up to `per_side` kept and `per_side` dropped sentences from the verifying modes, drawn at random."""

    kept, dropped = [], []

    for row in report["rows"]:
        if row.get("mode") not in VERIFIED_MODES or "error" in row:
            continue

        base = {"question": row["question"], "mode": row["mode"], "sources": row.get("sources")}
        kept += [{**base, "decision": "kept", "sentence": item["text"], "source": item.get("source")} for item in row.get("kept", [])]
        dropped += [{**base, "decision": "dropped", "sentence": item if isinstance(item, str) else item.get("text", "")} for item in row.get("dropped_sentences", [])]

    rng = random.Random(seed)
    chosen = rng.sample(kept, min(per_side, len(kept))) + rng.sample(dropped, min(per_side, len(dropped)))
    rng.shuffle(chosen)       # the labeller should not see which side a sentence is on

    return [{"id": i, **item, "supported_by_sources": None} for i, item in enumerate(chosen)]


def score(sheet: list[dict]) -> dict:
    """False-accept and false-reject rates from labelled rows (`supported_by_sources` true or false)."""

    labelled = [r for r in sheet if r.get("supported_by_sources") is not None]
    kept = [r for r in labelled if r["decision"] == "kept"]
    dropped = [r for r in labelled if r["decision"] == "dropped"]
    false_accept = sum(1 for r in kept if r["supported_by_sources"] is False)
    false_reject = sum(1 for r in dropped if r["supported_by_sources"] is True)

    def rate(k, n):
        return {"errors": k, "n": n, "rate": k / n if n else None, "interval": exact_interval(k, n) if n else None}

    return {"labelled": len(labelled), "false_accept": rate(false_accept, len(kept)), "false_reject": rate(false_reject, len(dropped))}


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("sample")
    a.add_argument("ladder")
    a.add_argument("--per-side", type=int, default=30)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    b = sub.add_parser("score")
    b.add_argument("sheet")
    args = parser.parse_args()

    if args.command == "sample":
        sheet = sample(json.load(open(args.ladder, encoding="utf-8")), args.per_side, args.seed)
        json.dump(sheet, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"{len(sheet)} sentences written to {args.out}; fill in supported_by_sources (true / false) for each")
        return

    result = score(json.load(open(args.sheet, encoding="utf-8")))

    for name in ("false_accept", "false_reject"):
        r = result[name]
        text = "-" if not r["n"] else f"{r['errors']}/{r['n']} = {100 * r['rate']:.1f}% (exact 95% interval {100 * r['interval'][0]:.0f}-{100 * r['interval'][1]:.0f}%)"
        print(f"{name:13s} {text}")


if __name__ == "__main__":
    main()
