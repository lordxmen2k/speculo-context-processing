"""Speculo: a persistent data structure for speculative context states.

A quantum-inspired DAG of superposition nodes (each holding N
candidate context slices with amplitudes) and collapsed nodes
(the result of measuring a superposition). Every operation
returns a new immutable root; old roots remain valid.

Public API:
    Speculo                  — the persistent DAG class
    Slice, Tier, Oracle      — the type system
    uniform_oracle, first_oracle, keyword_oracle,
        random_oracle, embedding_oracle — built-in oracles
    SpeculoError, EmptyStateError, etc. — error types
"""
from .core import (
    Speculo,
    SpeculoError,
    EmptyStateError,
    UnknownNodeError,
    InvalidAmplitudeError,
    NoCandidatesError,
    SliceNode,
    SuperpositionNode,
    CollapsedNode,
    OutcomeRecord,
)
from .types import (
    Slice,
    Tier,
    Oracle,
    NodeID,
    SliceID,
    NULL_NODE,
    NULL_SLICE,
)
from .oracle import (
    uniform_oracle,
    first_oracle,
    keyword_oracle,
    random_oracle,
    embedding_oracle,
)

__version__ = "0.0.1.dev0"

__all__ = [
    "Speculo",
    "Slice",
    "Tier",
    "Oracle",
    "NodeID",
    "SliceID",
    "NULL_NODE",
    "NULL_SLICE",
    "uniform_oracle",
    "first_oracle",
    "keyword_oracle",
    "random_oracle",
    "embedding_oracle",
    "SpeculoError",
    "EmptyStateError",
    "UnknownNodeError",
    "InvalidAmplitudeError",
    "NoCandidatesError",
    "SliceNode",
    "SuperpositionNode",
    "CollapsedNode",
    "OutcomeRecord",
]
