"""The systems compared by the long-conversation benchmark.

Each variant is a set of overrides of `settings` plus a way of building the
conversation history the LLM sees. Everything else (retrieval, generation,
verification, the document) is identical, so a difference in accuracy comes
from how the conversation is remembered and nothing else.

Baselines (no SnapRAG memory):
    no_memory        the last 3 turns only
    full_history     every turn, in full, newest first up to a token cap
    summary_buffer   a rolling LLM summary of old turns + the last 3 turns
                     (the MemGPT-style "summarize and buffer" baseline)

The reconstruction ladder (the paper's main figure; the conversation state
is off, so only snapshot recall differs):
    snapshot         rebuild from the snapshot alone, nothing checked
    evidence         + the document pages the snapshot cites
    verify           + each sentence checked against snapshot and pages
    archive          + checked against the chat record when still doubtful

SnapRAG:
    state_only       the verified conversation state, no snapshot recall
    full             verified reconstruction + conversation state

Ablations of `full`:
    full_fixed_lambda    memory weight fixed at 1.0 instead of the turn curve
    full_no_exemplars    snapshots do not see earlier snapshots
    full_fixed_interval  snapshot every 5 turns instead of the quality trigger
    full_fixed_k         no extra retrieval batches: fixed-k RAG, the AB-RAG ablation loop

Not run here: LCM, Mem0, A-MEM and LLMLingua. They are separate systems with
their own stores; comparing against them needs their code or a faithful
re-implementation, which this harness does not pretend to be.
"""

from dataclasses import dataclass, field

# Full-history cap, in tokens (estimated at 4 characters per token).
FULL_HISTORY_TOKENS = 30_000

_OFF = {"MEMORY_RECALL_MODE": "off", "MEMORY_STATE_ENABLED": False}
_LADDER = {"MEMORY_STATE_ENABLED": False}


@dataclass(frozen=True)
class Variant:

    name: str
    description: str
    settings: dict = field(default_factory=dict)
    # "snaprag": prompt_history + snapshots; "window": last turns only;
    # "summary": rolling summary + last turns.
    history: str = "snaprag"
    uses_snapshots: bool = True


def _v(name, description, settings=None, **kwargs):
    return Variant(name, description, settings or {}, **kwargs)


VARIANTS = {v.name: v for v in [
    _v("no_memory", "last 3 turns only", {**_OFF, "HISTORY_MAX_TOKENS": 0}, history="window", uses_snapshots=False),
    _v("full_history", f"every turn in full (cap {FULL_HISTORY_TOKENS} tokens)",
       {**_OFF, "HISTORY_TURNS": 0, "HISTORY_MAX_TOKENS": FULL_HISTORY_TOKENS, "HISTORY_ANSWER_CHARS": 10**8},
       history="window", uses_snapshots=False),
    _v("summary_buffer", "rolling LLM summary + last 3 turns", {**_OFF, "HISTORY_MAX_TOKENS": 0},
       history="summary", uses_snapshots=False),

    _v("snapshot", "reconstruct from the snapshot, unchecked", {**_LADDER, "MEMORY_RECALL_MODE": "snapshot"}),
    _v("evidence", "+ document pages, unchecked", {**_LADDER, "MEMORY_RECALL_MODE": "evidence"}),
    _v("verify", "+ checked against snapshot and pages", {**_LADDER, "MEMORY_RECALL_MODE": "verify"}),
    _v("archive", "+ checked against the chat record (verified reconstruction)", {**_LADDER, "MEMORY_RECALL_MODE": "archive"}),

    _v("state_only", "conversation state only", {"MEMORY_RECALL_MODE": "state", "MEMORY_STATE_ENABLED": True}),
    _v("full", "verified reconstruction + conversation state", {"MEMORY_RECALL_MODE": "archive", "MEMORY_STATE_ENABLED": True}),

    _v("full_fixed_lambda", "full, memory weight fixed at 1.0",
       {"MEMORY_RECALL_MODE": "archive", "MEMORY_STATE_ENABLED": True, "LAMBDA_FIXED": 1.0}),
    _v("full_no_exemplars", "full, no earlier snapshots as exemplars",
       {"MEMORY_RECALL_MODE": "archive", "MEMORY_STATE_ENABLED": True, "SNAPSHOT_EXEMPLARS": 0}),
    _v("full_fixed_interval", "full, snapshot every 5 turns",
       {"MEMORY_RECALL_MODE": "archive", "MEMORY_STATE_ENABLED": True, "CQM_FIXED_INTERVAL": 5,
        "CQM_CHECK_CONTRADICTION": False}),
    _v("full_fixed_k", "full, fixed-k retrieval (no extra batches)",
       {"MEMORY_RECALL_MODE": "archive", "MEMORY_STATE_ENABLED": True, "MAX_EXTRA_BATCHES": 0}),
]}

GROUPS = {
    "main": ["no_memory", "full_history", "summary_buffer", "archive", "full"],
    "baselines": ["no_memory", "full_history", "summary_buffer"],
    "ladder": ["snapshot", "evidence", "verify", "archive", "state_only", "full"],
    "ablations": ["full", "full_fixed_lambda", "full_no_exemplars", "full_fixed_interval", "full_fixed_k"],
    "all": list(VARIANTS),
}


def resolve(names: str) -> list[str]:
    """'main', 'ladder,ablations' or 'full,no_memory' -> variant names, no duplicates."""

    chosen = []

    for name in (n.strip() for n in names.split(",") if n.strip()):

        if name in GROUPS:
            chosen += GROUPS[name]
        elif name in VARIANTS:
            chosen.append(name)
        else:
            raise ValueError(f"unknown variant or group: {name} (variants: {', '.join(VARIANTS)}; groups: {', '.join(GROUPS)})")

    return list(dict.fromkeys(chosen))
