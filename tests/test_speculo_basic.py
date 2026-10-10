"""Unit tests for Speculo.

Covers the basic API: slice management, superposition,
measurement, error correction, refresh, eviction, branching,
tiers, and structural sharing invariants.
"""
import time
import pytest
import sys
sys.path.insert(0, "src")

from speculo import (
    Speculo, SpeculoError, EmptyStateError, UnknownNodeError,
    InvalidAmplitudeError, NoCandidatesError, Tier,
    uniform_oracle, first_oracle, keyword_oracle,
)


def make_speculo_with_slices(n: int = 3):
    """Helper: build a Speculo with `n` slices pre-populated."""
    s = Speculo()
    slice_ids = []
    for i in range(n):
        sid = s.add_slice(
            content=f"slice_{i}",
            summary=f"summary_{i}",
            length=10 * (i + 1),
        )
        slice_ids.append(sid)
    return s, slice_ids


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------


class TestConstruction:
    def test_empty(self):
        s = Speculo()
        assert s.is_empty()
        assert s.root_id() is None
        assert s.roots() == set()
        assert len(s) == 0

    def test_add_slice(self):
        s = Speculo()
        nid = s.add_slice("hello", summary="greeting", length=1)
        assert not s.is_empty()
        assert nid is not None
        assert s.is_slice(nid)
        assert s.get_slice(1).content == "hello"
        assert s.get_slice(1).summary == "greeting"
        assert s.get_slice(1).tier == Tier.HOT

    def test_add_multiple_slices(self):
        s, slice_ids = make_speculo_with_slices(5)
        assert len(s) == 5
        assert len(slice_ids) == 5
        assert all(s.is_slice(sid) for sid in slice_ids)


# ----------------------------------------------------------------------------
# Superposition
# ----------------------------------------------------------------------------


class TestSuperposition:
    def test_new_superposition(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        assert s.is_superposition(sp)
        candidates = s.get_candidates(sp)
        assert len(candidates) == 3
        assert abs(s.amplitude_sum(sp) - 1.0) < 1e-9

    def test_new_superposition_renormalizes(self):
        s, slice_ids = make_speculo_with_slices(2)
        # Pass amplitudes that don't sum to 1 — should be renormalized
        sp = s.new_superposition([(1, 0.4), (2, 0.6)])
        candidates = s.get_candidates(sp)
        # Sum should be 1.0
        assert abs(s.amplitude_sum(sp) - 1.0) < 1e-9
        # Each amplitude should be in (0, 1)
        amps = [a for _, a in candidates]
        assert all(0.0 < a < 1.0 for a in amps)

    def test_empty_superposition_rejected(self):
        s = Speculo()
        with pytest.raises(NoCandidatesError):
            s.new_superposition([])

    def test_invalid_amplitude_rejected(self):
        s, slice_ids = make_speculo_with_slices(2)
        with pytest.raises(InvalidAmplitudeError):
            s.new_superposition([(1, -0.1), (2, 0.5)])
        with pytest.raises(InvalidAmplitudeError):
            s.new_superposition([(1, 1.5), (2, 0.5)])

    def test_unknown_slice_rejected(self):
        s, slice_ids = make_speculo_with_slices(2)
        with pytest.raises(UnknownNodeError):
            s.new_superposition([(1, 0.5), (99, 0.5)])


# ----------------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------------


class TestMeasure:
    def test_measure_with_uniform_oracle(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, uniform_oracle, first_tokens="anything")
        assert s.is_collapsed(c)
        snap = s.resolve(c)
        assert snap[0]["kind"] == "collapsed"
        # Should have chosen one of the candidates
        assert snap[0]["chosen_slice_id"] in [1, 2, 3]

    def test_measure_with_first_oracle(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, first_oracle)
        # First oracle always picks the first candidate
        assert s.resolve(c)[0]["chosen_slice_id"] == 1
        assert s.resolve(c)[0]["confidence"] == 1.0

    def test_measure_with_keyword_oracle(self):
        s = Speculo()
        s.add_slice("Python is great.", summary="about Python", length=3)
        s.add_slice("Java is great.", summary="about Java", length=3)
        s.add_slice("Rust is fast.", summary="about Rust", length=3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, keyword_oracle, first_tokens="tell me about Python")
        # Keyword oracle should prefer slice 1 (Python)
        chosen = s.resolve(c)[0]["chosen_slice_id"]
        assert chosen == 1

    def test_collapsed_preserves_remaining(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, first_oracle)
        snap = s.resolve(c)
        # Remaining should have 2 candidates (1 chosen, 2 remain)
        assert len(snap[0]["remaining_candidates"]) == 2

    def test_measure_preserves_superposition(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        original_candidates = s.get_candidates(sp)
        s.measure(sp, first_oracle)
        # The superposition is still there
        assert s.get_candidates(sp) == original_candidates

    def test_measure_on_non_superposition_raises(self):
        s, slice_ids = make_speculo_with_slices(2)
        # slice_ids[0] is a SliceNode, not a SuperpositionNode
        with pytest.raises(UnknownNodeError):
            s.measure(slice_ids[0], first_oracle)


# ----------------------------------------------------------------------------
# Error correction
# ----------------------------------------------------------------------------


class TestErrorCorrect:
    def test_error_correct_returns_superposition(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, first_oracle)
        ec = s.error_correct(c, uniform_oracle, first_tokens="retry")
        assert s.is_superposition(ec)
        # The wrongly-chosen slice is excluded
        candidates = s.get_candidates(ec)
        chosen = s.resolve(c)[0]["chosen_slice_id"]
        assert chosen not in [cid for cid, _ in candidates]

    def test_error_correct_normalizes(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, first_oracle)
        ec = s.error_correct(c, uniform_oracle, first_tokens="retry")
        # Amplitudes should sum to 1
        assert abs(s.amplitude_sum(ec) - 1.0) < 1e-9

    def test_error_correct_no_remaining_raises(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.9), (2, 0.1)])
        c = s.measure(sp, first_oracle)
        # First oracle picks slice 1 with confidence 1.0
        # The remaining is just slice 2 — error correct should work
        ec = s.error_correct(c, uniform_oracle, first_tokens="retry")
        # Now error correct on ec with the only candidate left
        c2 = s.measure(ec, first_oracle)
        with pytest.raises(NoCandidatesError):
            s.error_correct(c2, uniform_oracle, first_tokens="retry2")


# ----------------------------------------------------------------------------
# Add candidate
# ----------------------------------------------------------------------------


class TestAddCandidate:
    def test_add_candidate_to_existing_superposition(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        s4 = s.add_slice("new slice", summary="new", length=1)
        sp2 = s.add_candidate(sp, 4, 0.4)
        candidates = s.get_candidates(sp2)
        assert len(candidates) == 4
        assert any(cid == 4 for cid, _ in candidates)
        # Sum should be 1.0
        assert abs(s.amplitude_sum(sp2) - 1.0) < 1e-9

    def test_add_candidate_invalid_amplitude(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        with pytest.raises(InvalidAmplitudeError):
            s.add_candidate(sp, 1, 1.5)

    def test_add_candidate_preserves_original(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        original = s.get_candidates(sp)
        s.add_candidate(sp, 1, 0.3)
        # Original is unchanged
        assert s.get_candidates(sp) == original


# ----------------------------------------------------------------------------
# Refresh
# ----------------------------------------------------------------------------


class TestRefresh:
    def test_refresh_amplitude(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        sp2 = s.refresh(sp, 2, 0.9)
        candidates = s.get_candidates(sp2)
        # Slice 2 should now have amplitude 0.9 (then renormalized)
        amps = {cid: a for cid, a in candidates}
        # After renormalizing 0.5 + 0.9 + 0.2 = 1.6, slice 2 = 0.9/1.6
        assert abs(amps[2] - 0.9 / 1.6) < 1e-9

    def test_refresh_preserves_others(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        sp2 = s.refresh(sp, 2, 0.3)  # same amplitude
        # Sum should still be 1.0
        assert abs(s.amplitude_sum(sp2) - 1.0) < 1e-9

    def test_refresh_invalid_slice(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        with pytest.raises(UnknownNodeError):
            s.refresh(sp, 99, 0.5)


# ----------------------------------------------------------------------------
# Evict stale (decoherence)
# ----------------------------------------------------------------------------


class TestEvictStale:
    def test_evict_removes_old(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        # All slices have similar created_at / last_observed
        # Set last_observed for slice 2 to a long time ago
        n2 = s._get(s._slice_index[2])
        with n2.lock:
            n2.last_observed = 0
        # Evict with a 1-second TTL — slice 2 is stale, others are fresh
        now = time.time()
        sp2 = s.evict_stale(sp, ttl_seconds=1.0, now=now)
        candidates = s.get_candidates(sp2)
        # Slice 2 should be evicted
        assert all(cid != 2 for cid, _ in candidates)

    def test_evict_all_stale_raises(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        # Set all to very old
        n1 = s._get(s._slice_index[1])
        n2 = s._get(s._slice_index[2])
        with n1.lock:
            n1.last_observed = 0
        with n2.lock:
            n2.last_observed = 0
        with pytest.raises(NoCandidatesError):
            s.evict_stale(sp, ttl_seconds=1.0, now=time.time())


# ----------------------------------------------------------------------------
# Branch
# ----------------------------------------------------------------------------


class TestBranch:
    def test_branch_preserves_candidates(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        b = s.branch(sp)
        assert s.get_candidates(b) == s.get_candidates(sp)

    def test_branch_returns_new_id(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        b = s.branch(sp)
        assert b != sp

    def test_branch_does_not_mutate_original(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        b = s.branch(sp)
        # Original still has its original candidates
        assert s.get_candidates(sp) == [(1, 0.5), (2, 0.5)]


# ----------------------------------------------------------------------------
# Collapse to (manual selection)
# ----------------------------------------------------------------------------


class TestCollapseTo:
    def test_collapse_to_picks_specific(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.collapse_to(sp, 3)
        assert s.resolve(c)[0]["chosen_slice_id"] == 3
        assert s.resolve(c)[0]["confidence"] == 1.0

    def test_collapse_to_invalid_slice(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        with pytest.raises(UnknownNodeError):
            s.collapse_to(sp, 99)


# ----------------------------------------------------------------------------
# Tier transitions
# ----------------------------------------------------------------------------


class TestTiers:
    def test_default_tier_is_hot(self):
        s = Speculo()
        nid = s.add_slice("x", summary="x", length=1)
        assert s.get_slice(1).tier == Tier.HOT

    def test_promote_to_hot(self):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        s.demote(1, Tier.WARM)
        s.promote(1)
        assert s.get_slice(1).tier == Tier.HOT

    def test_demote_to_warm_and_cold(self):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        s.demote(1, Tier.WARM)
        assert s.get_slice(1).tier == Tier.WARM
        s.demote(1, Tier.COLD)
        assert s.get_slice(1).tier == Tier.COLD


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------


class TestStats:
    def test_empty_stats(self):
        s = Speculo()
        st = s.stats()
        assert st["n_nodes"] == 0
        assert st["n_slices"] == 0
        assert st["n_superpositions"] == 0
        assert st["n_collapsed"] == 0

    def test_stats_after_operations(self):
        s, slice_ids = make_speculo_with_slices(3)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        s.measure(sp, first_oracle)
        s.branch(sp)
        st = s.stats()
        assert st["n_slices"] == 3
        assert st["n_superpositions"] >= 1
        assert st["n_collapsed"] >= 1


# ----------------------------------------------------------------------------
# Touch + outcomes
# ----------------------------------------------------------------------------


class TestTouchAndOutcomes:
    def test_touch_slice_updates_timestamp(self):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        before = s._get(s._slice_index[1]).last_observed
        time.sleep(0.01)
        s.touch_slice(1)
        after = s._get(s._slice_index[1]).last_observed
        assert after > before

    def test_record_outcome(self):
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        c = s.measure(sp, first_oracle)
        s.record_outcome(c, was_correct=True)
        outcomes = s.outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].was_correct is True
        assert outcomes[0].chosen_slice_id == 1


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------


class TestEdgeCases:
    def test_get_slice_unknown(self):
        s = Speculo()
        assert s.get_slice(99) is None

    def test_is_superposition_on_slice(self):
        s = Speculo()
        nid = s.add_slice("x")
        assert s.is_slice(nid)
        assert not s.is_superposition(nid)
        assert not s.is_collapsed(nid)

    def test_resolve_empty(self):
        s = Speculo()
        assert s.resolve(None) == []
        assert s.resolve(99) == []

    def test_measure_empty_oracle_raises(self):
        def bad_oracle(*args):
            return []
        s, slice_ids = make_speculo_with_slices(2)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        with pytest.raises(SpeculoError):
            s.measure(sp, bad_oracle)
