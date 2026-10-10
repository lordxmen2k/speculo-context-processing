"""Type definitions for Speculo.

The data structure has:
- SliceID, NodeID: 64-bit ints, never reused (Bw-tree invariant)
- Slice: a context slice (a sequence of tokens or KV blocks)
- Amplitude: a float in [0, 1], the Born-rule probability
- Tier: HOT/WARM/COLD storage tier
- Oracle: a callable that returns a probability distribution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# A unique, never-reused identifier for a node in the DAG.
NodeID = int

# A unique identifier for a slice (a context slice or KV block sequence).
# A SliceID is the "name" of a piece of content; a NodeID is the
# "location" in the mapping table.
SliceID = int

# Sentinels
NULL_NODE: NodeID = -1
NULL_SLICE: SliceID = -1


class Tier(Enum):
    """Storage tier for a slice.

    - HOT: full slice in HBM (µs latency)
    - WARM: full slice in CPU DRAM, summarized on demand (ms)
    - COLD: full slice in NVMe SSD, summary in DRAM (10s of ms)
    """
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


@dataclass
class Slice:
    """A concrete context slice.

    `content` is the actual data (token IDs, KV blocks, or any
    other payload). `summary` is an optional short description
    used by the oracle for quick comparison without loading the
    full slice.

    Slices are immutable once created. Mutations return new SliceIDs
    (which are NodeIDs for the SliceNode that holds this content).
    """
    slice_id: SliceID
    content: Any
    summary: Optional[str] = None
    length: int = 0  # length in tokens or blocks
    tier: Tier = Tier.HOT
    metadata: dict = field(default_factory=dict)


# An oracle: takes the first K tokens of the LLM's output + a
# list of (slice_id, summary) candidates, returns a probability
# distribution (a list of (slice_id, amplitude) pairs, summing to 1.0).
#
# The oracle is pluggable. Implementations can be:
# - Embedding-based: compare question embedding to summary embeddings
# - Learned: a small model trained on past question/speculation pairs
# - LLM-based: a small LLM that evaluates each candidate
# - Random: a baseline for testing
Oracle = Callable[[Any, list[tuple[SliceID, Optional[str]]]], list[tuple[SliceID, float]]]


# A "trivial" oracle: returns uniform distribution. Useful for
# testing, debugging, or when no real oracle is available.
def uniform_oracle(first_tokens: Any,
                   candidates: list[tuple[SliceID, Optional[str]]]
                   ) -> list[tuple[SliceID, float]]:
    """Return uniform probability over all candidates."""
    if not candidates:
        return []
    p = 1.0 / len(candidates)
    return [(cid, p) for cid, _ in candidates]


# A "first-candidate" oracle: always returns the first candidate
# with amplitude 1.0. Useful for testing the data structure
# without the oracle logic.
def first_oracle(first_tokens: Any,
                candidates: list[tuple[SliceID, Optional[str]]]
                ) -> list[tuple[SliceID, float]]:
    """Always pick the first candidate with amplitude 1.0."""
    if not candidates:
        return []
    return [(candidates[0][0], 1.0)]
