"""Speculo: a persistent DAG for speculative context states.

The data structure has three node types:
- SliceNode: a concrete context slice (immutable)
- SuperpositionNode: N candidate SliceNodes with amplitudes (sum=1.0)
- CollapsedNode: the result of measuring a SuperpositionNode

Operations (all return a new root; old roots remain valid):
- add_candidate(superposition, slice, amplitude) -> new_superposition
- measure(superposition, oracle, first_tokens) -> CollapsedNode
- error_correct(collapsed, oracle) -> new SuperpositionNode
- refresh(superposition, slice_id, new_amplitude) -> new_superposition
- evict_stale(superposition, ttl_seconds, now) -> new_superposition
- collapse_to(superposition, slice_id) -> CollapsedNode
- branch(superposition) -> new_superposition (clone with new ID)
- record_outcome(collapsed, was_correct) -> None (side effect)

Thread safety: per-Speculo mutation lock; reads are lock-free.

The quantum analogy (superposition, measurement, Born rule,
decoherence, error correction) is taken seriously as a data
structure design tool, not as a runtime trick.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .types import (
    Oracle, Slice, SliceID, Tier, NodeID, NULL_NODE, NULL_SLICE,
)


# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------


class SpeculoError(Exception):
    """Base error for Speculo operations."""


class EmptyStateError(SpeculoError):
    """Operation requires a non-empty state but got an empty root."""


class UnknownNodeError(SpeculoError):
    """Referenced NodeID or SliceID is not in the mapping."""


class InvalidAmplitudeError(SpeculoError):
    """Amplitude is not a valid probability (not in [0, 1], or
    sum-to-1 violated)."""


class NoCandidatesError(SpeculoError):
    """Operation requires at least one candidate but got none."""


# ----------------------------------------------------------------------------
# Node types
# ----------------------------------------------------------------------------


@dataclass
class SliceNode:
    """A concrete context slice (immutable).

    `content` is the actual data. `summary` is an optional short
    description used by the oracle. `tier` controls where the
    content lives (HOT/WARM/COLD).
    """
    node_id: NodeID
    slice_id: SliceID
    content: Any
    summary: Optional[str] = None
    length: int = 0
    tier: Tier = Tier.HOT
    refcount: int = 1
    created_at: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class SuperpositionNode:
    """N candidate SliceNodes with amplitudes (Born rule: sum=1.0).

    `candidates` is a list of (slice_id, amplitude) pairs. Amplitudes
    are calibrated by refresh() based on observed outcomes.
    `parent_refs` allows this node to be a candidate in a higher-level
    superposition (tree of speculations).
    """
    node_id: NodeID
    candidates: list  # list[tuple[SliceID, float]]
    parent_refs: list  # list[NodeID]
    created_at: float = field(default_factory=time.time)
    refcount: int = 1
    metadata: dict = field(default_factory=dict)

    def amplitude_sum(self) -> float:
        return sum(a for _, a in self.candidates)


@dataclass
class CollapsedNode:
    """The result of measuring a SuperpositionNode.

    Holds the chosen slice, the confidence, the source
    superposition, and (importantly) the unchosen candidates as
    children. This is what enables error correction: if the
    collapse was wrong, we can re-enter superposition with the
    remaining candidates.
    """
    node_id: NodeID
    chosen_slice_id: SliceID
    confidence: float
    source_superposition: NodeID
    remaining_candidates: list  # list[tuple[SliceID, float]]
    oracle_used: Optional[Any] = None
    refcount: int = 1
    created_at: float = field(default_factory=time.time)


@dataclass
class OutcomeRecord:
    """Records the outcome of a collapse (for calibration).

    `was_correct` is True if the LLM's full output matched the
    chosen slice's content. Over time, this data trains the
    oracle (P0-XX: online learning).
    """
    collapsed_id: NodeID
    chosen_slice_id: SliceID
    was_correct: bool
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# The Speculo class
# ----------------------------------------------------------------------------


class Speculo:
    """A persistent DAG of speculative context states.

    The class holds a single mapping table (NodeID -> node) and
    tracks one or more "live roots" — the currently-active
    speculation references. Operations return new roots; old
    roots remain valid until their refcount drops to 0 and all
    live readers finish.

    Thread safety:
    - Mutating ops (add_candidate, measure, error_correct, etc.)
      hold the per-Speculo `_mutation_lock` for their duration.
    - Reads (resolve, get_candidates, is_empty, stats) are
      lock-free; they return a snapshot.

    The Rust port replaces the mutation lock with per-node CAS.
    """

    def __init__(self):
        self._mapping: dict[NodeID, Any] = {}
        self._slice_index: dict[SliceID, NodeID] = {}  # slice_id -> node_id
        self._deferred_free: list[NodeID] = []
        self._next_id: int = 1
        self._next_slice_id: int = 1
        self._id_lock = threading.Lock()
        self._mutation_lock = threading.Lock()
        self._mutation_count: int = 0
        self._roots: set[NodeID] = set()
        self._outcomes: list[OutcomeRecord] = []  # for oracle calibration
        self._outcomes_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API: queries (read-only, lock-free)
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return len(self._mapping) == 0

    def root_id(self) -> Optional[NodeID]:
        if not self._roots:
            return None
        return next(iter(self._roots))

    def roots(self) -> set[NodeID]:
        return set(self._roots)

    def refcount_of(self, nid: NodeID) -> int:
        n = self._get(nid)
        if n is None:
            return 0
        return getattr(n, "refcount", 0)

    def get_slice(self, slice_id: SliceID) -> Optional[Slice]:
        nid = self._slice_index.get(slice_id)
        if nid is None:
            return None
        n = self._get(nid)
        if not isinstance(n, SliceNode):
            return None
        return Slice(
            slice_id=n.slice_id,
            content=n.content,
            summary=n.summary,
            length=n.length,
            tier=n.tier,
        )

    def get_candidates(self, superposition_id: NodeID) -> list:
        """Return the (slice_id, amplitude) candidates in a superposition."""
        n = self._get(superposition_id)
        if n is None or not isinstance(n, SuperpositionNode):
            raise UnknownNodeError(f"not a superposition: {superposition_id}")
        return list(n.candidates)

    def is_superposition(self, nid: NodeID) -> bool:
        n = self._get(nid)
        return isinstance(n, SuperpositionNode)

    def is_collapsed(self, nid: NodeID) -> bool:
        n = self._get(nid)
        return isinstance(n, CollapsedNode)

    def is_slice(self, nid: NodeID) -> bool:
        n = self._get(nid)
        return isinstance(n, SliceNode)

    def amplitude_sum(self, superposition_id: NodeID) -> float:
        """Return the sum of amplitudes in a superposition (should be 1.0)."""
        return self.get_candidates(superposition_id) and sum(
            a for _, a in self.get_candidates(superposition_id)
        ) or 0.0

    def outcomes(self) -> list:
        """Return a copy of the outcome records (for oracle calibration)."""
        with self._outcomes_lock:
            return list(self._outcomes)

    def stats(self) -> dict:
        """Return a snapshot of internal statistics."""
        n_slices = sum(1 for n in self._mapping.values() if isinstance(n, SliceNode))
        n_super = sum(1 for n in self._mapping.values() if isinstance(n, SuperpositionNode))
        n_coll = sum(1 for n in self._mapping.values() if isinstance(n, CollapsedNode))
        by_tier = {t: 0 for t in Tier}
        for n in self._mapping.values():
            if isinstance(n, SliceNode):
                by_tier[n.tier] += 1
        return {
            "n_nodes": len(self._mapping),
            "n_slices": n_slices,
            "n_superpositions": n_super,
            "n_collapsed": n_coll,
            "n_roots": len(self._roots),
            "deferred_free": len(self._deferred_free),
            "mutations": self._mutation_count,
            "next_id": self._next_id,
            "tier_counts": {t.value: c for t, c in by_tier.items()},
            "n_outcomes": len(self._outcomes),
        }

    def resolve(self, root: Optional[NodeID]) -> list:
        """Return a snapshot of the path from root to its source.

        For a SuperpositionNode: the candidates.
        For a CollapsedNode: the chosen slice, confidence, and
        remaining candidates.
        For a SliceNode: the slice content.
        """
        if root is None or root == NULL_NODE:
            return []
        n = self._get(root)
        if n is None:
            return []
        if isinstance(n, SliceNode):
            return [{"kind": "slice", "slice_id": n.slice_id, "tier": n.tier,
                     "summary": n.summary, "length": n.length}]
        if isinstance(n, SuperpositionNode):
            return [{"kind": "superposition", "node_id": n.node_id,
                     "candidates": list(n.candidates),
                     "amplitude_sum": n.amplitude_sum(),
                     "created_at": n.created_at}]
        if isinstance(n, CollapsedNode):
            return [{"kind": "collapsed", "node_id": n.node_id,
                     "chosen_slice_id": n.chosen_slice_id,
                     "confidence": n.confidence,
                     "remaining_candidates": list(n.remaining_candidates)}]
        return []

    # ------------------------------------------------------------------
    # Public API: structural mutations (hold the mutation lock)
    # ------------------------------------------------------------------

    def add_slice(self, content: Any, summary: Optional[str] = None,
                  length: int = 0, tier: Tier = Tier.HOT) -> SliceID:
        """Add a new SliceNode. Returns its SliceID.

        The slice is also added to live roots (so it's
        immediately usable). The internal NodeID can be
        retrieved via the slice_index.
        """
        with self._mutation_lock:
            slice_id = self._alloc_slice_id()
            node_id = self._alloc_id()
            node = SliceNode(
                node_id=node_id,
                slice_id=slice_id,
                content=content,
                summary=summary,
                length=length,
                tier=tier,
                refcount=1,
            )
            self._put(node)
            self._slice_index[slice_id] = node_id
            self._roots.add(node_id)
            self._bump_mutation()
            return slice_id

    def slice_to_node(self, slice_id: SliceID) -> Optional[NodeID]:
        """Get the NodeID for a SliceID (the internal mapping)."""
        return self._slice_index.get(slice_id)

    def add_candidate(self, superposition_id: NodeID, slice_id: SliceID,
                      amplitude: float) -> NodeID:
        """Add a candidate to a SuperpositionNode. Returns a new
        SuperpositionNode (the old one is unchanged).

        Amplitudes are renormalized so the sum remains 1.0.
        """
        if not 0.0 <= amplitude <= 1.0:
            raise InvalidAmplitudeError(
                f"amplitude must be in [0, 1], got {amplitude}"
            )
        with self._mutation_lock:
            n = self._get(superposition_id)
            if n is None or not isinstance(n, SuperpositionNode):
                raise UnknownNodeError(f"not a superposition: {superposition_id}")
            if slice_id not in self._slice_index:
                raise UnknownNodeError(f"unknown slice: {slice_id}")
            # Bump refcount on the slice (it's now referenced by
            # the new superposition).
            slice_node_id = self._slice_index[slice_id]
            self._bump_refcount(slice_node_id)
            # Bump refcount on the old superposition (new
            # references the old).
            self._bump_refcount(superposition_id)
            # Build new candidate list
            new_candidates = list(n.candidates) + [(slice_id, amplitude)]
            # Renormalize
            total = sum(a for _, a in new_candidates)
            if total > 0:
                new_candidates = [(c, a / total) for c, a in new_candidates]
            new_id = self._alloc_id()
            new_node = SuperpositionNode(
                node_id=new_id,
                candidates=new_candidates,
                parent_refs=[superposition_id],
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def new_superposition(self, candidates: list) -> NodeID:
        """Create a new SuperpositionNode from a list of
        (slice_id, amplitude) pairs. Amplitudes are renormalized.

        Returns the new node's ID.
        """
        if not candidates:
            raise NoCandidatesError("superposition must have at least one candidate")
        # Validate
        for cid, amp in candidates:
            if not 0.0 <= amp <= 1.0:
                raise InvalidAmplitudeError(
                    f"amplitude must be in [0, 1], got {amp} for {cid}"
                )
            if cid not in self._slice_index:
                raise UnknownNodeError(f"unknown slice: {cid}")
        # Renormalize
        total = sum(a for _, a in candidates)
        if total <= 0:
            raise InvalidAmplitudeError("amplitudes must sum to > 0")
        normed = [(c, a / total) for c, a in candidates]
        with self._mutation_lock:
            # Bump refcount on each slice
            for cid, _ in normed:
                self._bump_refcount(self._slice_index[cid])
            new_id = self._alloc_id()
            new_node = SuperpositionNode(
                node_id=new_id,
                candidates=normed,
                parent_refs=[],
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def measure(self, superposition_id: NodeID, oracle: Oracle,
                first_tokens: Any = None) -> NodeID:
        """Apply the oracle to collapse a superposition to one slice.

        Returns a CollapsedNode ID. The remaining candidates are
        preserved as children (for error correction).
        """
        with self._mutation_lock:
            n = self._get(superposition_id)
            if n is None or not isinstance(n, SuperpositionNode):
                raise UnknownNodeError(f"not a superposition: {superposition_id}")
            if not n.candidates:
                raise NoCandidatesError(f"superposition {superposition_id} has no candidates")
            # Build the candidate list for the oracle
            oracle_input = []
            for cid, _ in n.candidates:
                slice_node = self._get(self._slice_index[cid])
                if slice_node is not None:
                    oracle_input.append((cid, slice_node.summary))
            # Apply the oracle
            distribution = oracle(first_tokens, oracle_input)
            if not distribution:
                raise SpeculoError("oracle returned empty distribution")
            # Validate the distribution
            total = sum(a for _, a in distribution)
            if total <= 0:
                raise SpeculoError("oracle returned zero-amplitude distribution")
            normed = [(c, a / total) for c, a in distribution]
            # Find the chosen candidate (highest amplitude)
            chosen_id = max(normed, key=lambda x: x[1])[0]
            confidence = max(a for _, a in normed)
            # Build the CollapsedNode
            # Remaining candidates = all candidates minus chosen
            remaining = [(c, a) for c, a in n.candidates if c != chosen_id]
            # Bump refcounts
            self._bump_refcount(superposition_id)
            if chosen_id in self._slice_index:
                self._bump_refcount(self._slice_index[chosen_id])
            for cid, _ in remaining:
                if cid in self._slice_index:
                    self._bump_refcount(self._slice_index[cid])
            new_id = self._alloc_id()
            new_node = CollapsedNode(
                node_id=new_id,
                chosen_slice_id=chosen_id,
                confidence=confidence,
                source_superposition=superposition_id,
                remaining_candidates=remaining,
                oracle_used=oracle,
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def error_correct(self, collapsed_id: NodeID, oracle: Oracle,
                      first_tokens: Any = None) -> NodeID:
        """The collapse was wrong. Re-enter superposition with the
        remaining candidates. The wrongly-chosen slice is excluded.

        If no candidates remain, raises NoCandidatesError.
        """
        with self._mutation_lock:
            n = self._get(collapsed_id)
            if n is None or not isinstance(n, CollapsedNode):
                raise UnknownNodeError(f"not a collapsed node: {collapsed_id}")
            if not n.remaining_candidates:
                raise NoCandidatesError(
                    f"no remaining candidates for {collapsed_id}; "
                    "fall back to full-context processing"
                )
            # Renormalize remaining candidates
            total = sum(a for _, a in n.remaining_candidates)
            if total <= 0:
                raise NoCandidatesError(
                    f"remaining candidates for {collapsed_id} have zero amplitude"
                )
            new_candidates = [(c, a / total) for c, a in n.remaining_candidates]
            # Bump refcounts
            for cid, _ in new_candidates:
                if cid in self._slice_index:
                    self._bump_refcount(self._slice_index[cid])
            self._bump_refcount(collapsed_id)
            new_id = self._alloc_id()
            new_node = SuperpositionNode(
                node_id=new_id,
                candidates=new_candidates,
                parent_refs=[collapsed_id],
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def refresh(self, superposition_id: NodeID, slice_id: SliceID,
                new_amplitude: float) -> NodeID:
        """Update the amplitude of a candidate. Returns a new
        SuperpositionNode. Other candidates' amplitudes are
        unchanged; the sum is renormalized.
        """
        if not 0.0 <= new_amplitude <= 1.0:
            raise InvalidAmplitudeError(
                f"amplitude must be in [0, 1], got {new_amplitude}"
            )
        with self._mutation_lock:
            n = self._get(superposition_id)
            if n is None or not isinstance(n, SuperpositionNode):
                raise UnknownNodeError(f"not a superposition: {superposition_id}")
            # Find and update the candidate
            new_candidates = []
            found = False
            for cid, amp in n.candidates:
                if cid == slice_id:
                    new_candidates.append((cid, new_amplitude))
                    found = True
                else:
                    new_candidates.append((cid, amp))
            if not found:
                raise UnknownNodeError(
                    f"slice {slice_id} not in superposition {superposition_id}"
                )
            # Renormalize
            total = sum(a for _, a in new_candidates)
            if total > 0:
                new_candidates = [(c, a / total) for c, a in new_candidates]
            self._bump_refcount(superposition_id)
            new_id = self._alloc_id()
            new_node = SuperpositionNode(
                node_id=new_id,
                candidates=new_candidates,
                parent_refs=[superposition_id],
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def evict_stale(self, superposition_id: NodeID, ttl_seconds: float,
                    now: Optional[float] = None) -> NodeID:
        """Remove candidates whose last_observed is older than ttl.

        Returns a new SuperpositionNode with the stale candidates
        removed. If all candidates are stale, raises NoCandidatesError.
        """
        if now is None:
            now = time.time()
        with self._mutation_lock:
            n = self._get(superposition_id)
            if n is None or not isinstance(n, SuperpositionNode):
                raise UnknownNodeError(f"not a superposition: {superposition_id}")
            threshold = now - ttl_seconds
            new_candidates = []
            for cid, amp in n.candidates:
                slice_node = self._get(self._slice_index.get(cid, NULL_NODE))
                if slice_node is None:
                    continue
                if slice_node.last_observed < threshold:
                    # Decrement refcount (we're dropping this slice)
                    self._decrement_refcount(slice_node.node_id)
                    continue
                new_candidates.append((cid, amp))
            if not new_candidates:
                raise NoCandidatesError(
                    f"all candidates in {superposition_id} are stale"
                )
            total = sum(a for _, a in new_candidates)
            if total > 0:
                new_candidates = [(c, a / total) for c, a in new_candidates]
            self._bump_refcount(superposition_id)
            new_id = self._alloc_id()
            new_node = SuperpositionNode(
                node_id=new_id,
                candidates=new_candidates,
                parent_refs=[superposition_id],
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def collapse_to(self, superposition_id: NodeID, slice_id: SliceID) -> NodeID:
        """Manually pick a candidate (bypass the oracle)."""
        with self._mutation_lock:
            n = self._get(superposition_id)
            if n is None or not isinstance(n, SuperpositionNode):
                raise UnknownNodeError(f"not a superposition: {superposition_id}")
            if slice_id not in [c for c, _ in n.candidates]:
                raise UnknownNodeError(
                    f"slice {slice_id} not in superposition {superposition_id}"
                )
            remaining = [(c, a) for c, a in n.candidates if c != slice_id]
            self._bump_refcount(superposition_id)
            self._bump_refcount(self._slice_index[slice_id])
            for cid, _ in remaining:
                if cid in self._slice_index:
                    self._bump_refcount(self._slice_index[cid])
            new_id = self._alloc_id()
            new_node = CollapsedNode(
                node_id=new_id,
                chosen_slice_id=slice_id,
                confidence=1.0,  # manual = full confidence
                source_superposition=superposition_id,
                remaining_candidates=remaining,
                oracle_used=None,
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def branch(self, superposition_id: NodeID) -> NodeID:
        """Fork a superposition. Returns a new SuperpositionNode
        with the same candidates and a new ID. The original is
        unchanged.
        """
        with self._mutation_lock:
            n = self._get(superposition_id)
            if n is None or not isinstance(n, SuperpositionNode):
                raise UnknownNodeError(f"not a superposition: {superposition_id}")
            self._bump_refcount(superposition_id)
            for cid, _ in n.candidates:
                if cid in self._slice_index:
                    self._bump_refcount(self._slice_index[cid])
            new_id = self._alloc_id()
            new_node = SuperpositionNode(
                node_id=new_id,
                candidates=list(n.candidates),
                parent_refs=[superposition_id],
                refcount=1,
            )
            self._put(new_node)
            self._roots.add(new_id)
            self._bump_mutation()
            return new_id

    def record_outcome(self, collapsed_id: NodeID, was_correct: bool,
                       metadata: Optional[dict] = None) -> None:
        """Record the outcome of a collapse. Used for oracle
        calibration (P0-XX: online learning).

        This is a side-effect, not a structural mutation; the
        DAG is unchanged.
        """
        with self._outcomes_lock:
            self._outcomes.append(OutcomeRecord(
                collapsed_id=collapsed_id,
                chosen_slice_id=self._get(collapsed_id).chosen_slice_id
                if self.is_collapsed(collapsed_id) else NULL_SLICE,
                was_correct=was_correct,
                metadata=metadata or {},
            ))

    def touch_slice(self, slice_id: SliceID) -> None:
        """Update the last_observed timestamp of a slice. Used by
        the cache layer to mark slices as recently used.
        """
        nid = self._slice_index.get(slice_id)
        if nid is None:
            return
        n = self._get(nid)
        if isinstance(n, SliceNode):
            with n.lock:
                n.last_observed = time.time()

    def promote(self, slice_id: SliceID) -> None:
        """Move a slice to the HOT tier."""
        nid = self._slice_index.get(slice_id)
        if nid is None:
            raise UnknownNodeError(f"unknown slice: {slice_id}")
        with self._mutation_lock:
            n = self._get(nid)
            if isinstance(n, SliceNode):
                with n.lock:
                    n.tier = Tier.HOT
                self._bump_mutation()

    def demote(self, slice_id: SliceID, tier: Tier) -> None:
        """Move a slice to WARM or COLD tier."""
        if tier == Tier.HOT:
            raise SpeculoError("use promote() to move to HOT")
        nid = self._slice_index.get(slice_id)
        if nid is None:
            raise UnknownNodeError(f"unknown slice: {slice_id}")
        with self._mutation_lock:
            n = self._get(nid)
            if isinstance(n, SliceNode):
                with n.lock:
                    n.tier = tier
                self._bump_mutation()

    def snapshot(self, root: NodeID) -> NodeID:
        """Bump the refcount of a root to keep it alive.

        Returns the same NodeID. The caller should store the
        returned ID if they want to ensure the root stays in
        the live set.
        """
        if root is None or root == NULL_NODE:
            return NULL_NODE
        with self._mutation_lock:
            self._bump_refcount(root)
            self._roots.add(root)
            self._bump_mutation()
            return root

    def drop_root(self, root: NodeID) -> None:
        """Stop tracking `root` as a live root. Decrements refcount
        of the root and walks down its chain.
        """
        with self._mutation_lock:
            if root not in self._roots:
                return
            self._roots.discard(root)
            self._decrement_refcount_chain(root)
            self._bump_mutation()

    def collect_garbage(self) -> int:
        """Free all nodes in the deferred-free list. Returns count freed."""
        n = 0
        for nid in self._deferred_free:
            if nid in self._mapping:
                # Also remove from slice_index if it's a slice
                node = self._mapping[nid]
                if isinstance(node, SliceNode):
                    self._slice_index.pop(node.slice_id, None)
                del self._mapping[nid]
                n += 1
        self._deferred_free.clear()
        return n

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alloc_id(self) -> NodeID:
        with self._id_lock:
            nid = self._next_id
            self._next_id += 1
            return nid

    def _alloc_slice_id(self) -> SliceID:
        with self._id_lock:
            sid = self._next_slice_id
            self._next_slice_id += 1
            return sid

    def _put(self, node) -> None:
        self._mapping[node.node_id] = node

    def _get(self, nid: NodeID):
        return self._mapping.get(nid)

    def _bump_refcount(self, nid: NodeID) -> None:
        n = self._get(nid)
        if n is None:
            return
        if isinstance(n, SliceNode):
            with n.lock:
                n.refcount += 1
        else:
            n.refcount += 1

    def _decrement_refcount(self, nid: NodeID) -> None:
        n = self._get(nid)
        if n is None:
            return
        if isinstance(n, SliceNode):
            with n.lock:
                n.refcount -= 1
                if n.refcount <= 0:
                    self._deferred_free.append(nid)
        else:
            n.refcount -= 1
            if n.refcount <= 0:
                self._deferred_free.append(nid)

    def _decrement_refcount_chain(self, nid: NodeID) -> None:
        """Decrement refcount of `nid` and walk up its parent chain."""
        cur_id = nid
        visited = set()
        while cur_id is not None and cur_id not in visited:
            visited.add(cur_id)
            cur = self._get(cur_id)
            if cur is None:
                break
            old_refcount = getattr(cur, "refcount", 0)
            self._decrement_refcount(cur_id)
            if old_refcount > 1:
                # Still referenced; don't walk up
                break
            # Walk up to the first parent
            parents = getattr(cur, "parent_refs", [])
            if not parents:
                break
            cur_id = parents[0]

    def _bump_mutation(self) -> None:
        self._mutation_count += 1

    def __len__(self) -> int:
        return len(self._mapping)
