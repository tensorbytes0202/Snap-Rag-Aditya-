"""Builds the turn script of a long conversation from spec.json.

The conversation mixes four kinds of turns:

    doc         a long document question (the filler that fills the context)
    plant       the user states something about themselves or their task
    correction  the user later changes a planted value
    probe       a question whose answer is a planted fact (recall), a corrected
                one (supersede), an earlier turn (reference), or nothing at
                all (absent, trap: the right answer is "you have not told me")

Every probe has a distance: how many turns after its source turn it is asked.
Distances come from four buckets (near, mid, far, very far), scaled down for
short scripts, so accuracy can be plotted against distance, which is the
memory-retention curve the paper needs.

The same (spec, turns, seed) always gives the same script.
"""

import json
import os
import random

SPEC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec.json")


def load_spec(path: str | None = None) -> dict:

    with open(path or SPEC_PATH, encoding="utf-8") as f:
        return json.load(f)


def _scaled_buckets(spec: dict, turns: int) -> dict[str, tuple[int, int]]:
    """Distance ranges, shrunk so the longest still fits in a short script."""

    longest = max(hi for _, hi in spec["distance_buckets"].values())
    factor = min(1.0, max(0.1, (turns - 5) / (longest + 10)))

    return {
        name: (max(2, int(lo * factor)), max(3, int(hi * factor)))
        for name, (lo, hi) in spec["distance_buckets"].items()
    }


class _Slots:
    """Which turns are taken."""

    def __init__(self, turns: int, rng: random.Random):
        self.turns, self.rng, self.taken = turns, rng, {}

    def free(self, turn: int) -> bool:
        return 1 <= turn <= self.turns and turn not in self.taken

    def put(self, turn: int, item: dict) -> dict:
        assert self.free(turn), f"turn {turn} is not free"
        self.taken[turn] = item
        return item

    def pair(self, distance: int, first_from: int = 2, gap_before: int = 0):
        """(source turn, probe turn) with probe = source + distance, both free;
        the source turn leaves `gap_before` more turns before the probe."""

        last_source = self.turns - distance - gap_before

        if last_source < first_from:
            raise ValueError(f"a script of {self.turns} turns is too short for distance {distance}")

        for _ in range(500):
            source = self.rng.randint(first_from, last_source)
            if self.free(source) and self.free(source + gap_before + distance):
                return source, source + gap_before + distance

        raise ValueError("no free slots left for a probe; use more turns")

    def any_free(self, lo: int, hi: int) -> int:

        candidates = [t for t in range(lo, hi + 1) if self.free(t)]

        if not candidates:
            raise ValueError("no free slots left; use more turns")

        return self.rng.choice(candidates)


def build_script(spec: dict, turns: int = 100, seed: int = 0) -> list[dict]:
    """The conversation as a list of turn dicts, turn 1 first."""

    rng = random.Random(seed)
    slots = _Slots(turns, rng)
    buckets = _scaled_buckets(spec, turns)

    def distance(bucket: str) -> int:
        lo, hi = buckets[bucket]
        return rng.randint(lo, hi)

    def probe_item(probe_id, text, kind, bucket, dist, sources, **extra):
        return {"kind": "probe", "probe_id": probe_id, "text": text, "probe_kind": kind, "bucket": bucket,
                "distance": dist, "source_turns": sources, **extra}

    # Facts the user plants, asked about once each.
    for plant in spec["plants"]:
        dist = distance(plant["bucket"])
        source, probe = slots.pair(dist)
        slots.put(source, {"kind": "plant", "plant_id": plant["id"], "text": plant["statement"]})
        slots.put(probe, probe_item(plant["id"], plant["probe"], plant["kind"], plant["bucket"], dist, [source],
                                    expect_all=plant["expect_all"], gold=plant["gold"]))

    # Facts the user plants and later corrects; the distance counts from the correction.
    for item in spec.get("corrections", []):
        dist = distance(item["bucket"])

        for _ in range(200):
            gap = rng.randint(*item["gap"])
            source, probe = slots.pair(dist, gap_before=gap)
            if slots.free(source + gap):
                break
        else:
            raise ValueError("no free slots left for a correction; use more turns")

        correction = source + gap
        slots.put(source, {"kind": "plant", "plant_id": item["id"], "text": item["statement"]})
        slots.put(correction, {"kind": "correction", "plant_id": item["id"], "text": item["correction"]})
        slots.put(probe, probe_item(item["id"], item["probe"], item["kind"], item["bucket"], probe - correction,
                                    [source, correction], expect_all=item["expect_all"], stale=item.get("stale", []),
                                    gold=item["gold"]))

    # Questions with no answer in the conversation, asked in its second half.
    for item in spec.get("absent", []) + spec.get("traps", []):
        slot = slots.any_free(turns // 2, turns)
        slots.put(slot, probe_item(item["id"], item["question"], item["kind"], None, None, [],
                                   ok_cues=item.get("ok_cues", []), forbid=item.get("forbid", []), gold=item["gold"]))

    # "Which topic did I ask you about at turn N?": the source is a document turn.
    reference_slots = []
    refs = spec.get("references")

    if refs:
        for bucket in refs["bucket_order"][: refs["count"]]:
            dist = distance(bucket)
            source, probe = slots.pair(dist, first_from=3)
            slots.put(source, {"kind": "doc", "reference": True})
            slots.put(probe, {"kind": "probe", "placeholder": True})      # held until the topic is known
            reference_slots.append((source, probe, bucket, dist))

    def fill_doc(item: dict, topic: dict, lead: str) -> None:
        item.update({
            "text": spec["doc_template"].format(lead=lead, topic=topic["topic"]),
            "topic_id": topic["id"], "keyword": topic["keyword"], "gold_pages": topic.get("gold_pages"),
        })

    # A referenced turn needs a topic whose keyword cannot be confused with anything the user
    # plants (a short, generic keyword like "duty" would match almost any answer). Each such
    # topic is asked about exactly once, so the reference is unambiguous.
    blob = " ".join(
        [p["statement"] + p["gold"] for p in spec["plants"]]
        + [c["statement"] + c["correction"] + c["gold"] for c in spec.get("corrections", [])]
    ).lower()
    candidates = [t for t in spec["topics"] if len(t["keyword"]) >= 8 and t["keyword"].lower() not in blob]
    rng.shuffle(candidates)

    if len(candidates) < len(reference_slots):
        raise ValueError("not enough distinct topics for the reference probes")

    topic_of = {}

    for (source, _, _, _), topic in zip(reference_slots, candidates):
        fill_doc(slots.taken[source], topic, rng.choice(spec["doc_leads"]))
        topic_of[source] = topic

    # Everything else is document filler, drawn from a shuffled topic list.
    order = [t for t in spec["topics"] if t["id"] not in {topic["id"] for topic in topic_of.values()}]
    rng.shuffle(order)
    position = 0

    for turn in range(1, turns + 1):

        if turn in slots.taken:
            continue

        # The pass number shifts the lead, so a topic asked again reads differently.
        lead = spec["doc_leads"][(position % len(order) + position // len(order)) % len(spec["doc_leads"])]

        fill_doc(slots.taken.setdefault(turn, {"kind": "doc"}), order[position % len(order)], lead)
        position += 1

    for source, probe, bucket, dist in reference_slots:
        topic = topic_of[source]
        slots.taken[probe] = (probe_item(
            f"ref-{source}", refs["question"].format(turn=source), "reference", bucket, dist, [source],
            expect_all=[topic["keyword"]], gold=f"You asked me to explain {topic['topic']}.",
        ))

    script = []

    for turn in range(1, turns + 1):
        script.append({"turn": turn, **slots.taken[turn]})

    return script


def probes(script: list[dict]) -> list[dict]:

    return [item for item in script if item["kind"] == "probe"]
