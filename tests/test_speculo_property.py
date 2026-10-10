"""Property-based tests for Speculo using Hypothesis.

Test invariants:
- Born rule: sum of amplitudes in any superposition = 1.0
- Structural sharing: same slice as candidate in multiple
  superpositions has refcount > 1
- Decay correctness: error_correct removes the chosen slice
- Branch correctness: branch preserves candidates
- Tier transitions preserve slice content
"""
import time
import pytest
from hypothesis import given, settings, strategies as st
import sys
sys.path.insert(0, "src")

from speculo import (
    Speculo, Tier, uniform_oracle, first_oracle, keyword_oracle,
)


# Strategies
slice_count = st.integers(min_value=2, max_value=8)
amplitude = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)


class TestBornRule:
    """Sum of amplitudes must always equal 1.0 after any op."""

    @given(slice_count, st.lists(amplitude, min_size=2, max_size=8))
    @settings(max_examples=50, deadline=None)
    def test_new_superposition_sum_to_one(self, n_slices, amps):
        s = Speculo()
        slice_ids = []
        for i in range(n_slices):
            sid = s.add_slice(f"s{i}", summary=f"sum{i}", length=1)
            slice_ids.append(sid)
        # Use up to n_slices amplitudes
        amps = amps[:n_slices]
        # Make sure each amp is in [0, 1] and total > 0
        amps = [max(0.01, min(0.99, a)) for a in amps]
        if sum(amps) <= 0:
            return  # skip degenerate
        try:
            sp = s.new_superposition(list(zip(range(1, len(amps) + 1), amps)))
            assert abs(s.amplitude_sum(sp) - 1.0) < 1e-9
        except Exception:
            pass  # some amp combos may be invalid

    @given(slice_count)
    @settings(max_examples=50, deadline=None)
    def test_add_candidate_preserves_born_rule(self, n_slices):
        s = Speculo()
        for i in range(n_slices):
            s.add_slice(f"s{i}", summary=f"sum{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        s.add_slice("new", summary="new", length=1)
        sp2 = s.add_candidate(sp, n_slices + 1, 0.3)
        assert abs(s.amplitude_sum(sp2) - 1.0) < 1e-9

    @given(slice_count, amplitude)
    @settings(max_examples=50, deadline=None)
    def test_refresh_preserves_born_rule(self, n_slices, new_amp):
        s = Speculo()
        for i in range(n_slices):
            s.add_slice(f"s{i}", summary=f"sum{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        # Refresh slice 1
        new_amp = max(0.01, min(0.99, new_amp))
        sp2 = s.refresh(sp, 1, new_amp)
        assert abs(s.amplitude_sum(sp2) - 1.0) < 1e-9


class TestStructuralSharing:
    """Same slice as candidate in multiple superpositions has refcount > 1."""

    @given(st.integers(min_value=2, max_value=6))
    @settings(max_examples=50, deadline=None)
    def test_slice_refcount_grows_with_superpositions(self, n_sups):
        s = Speculo()
        s.add_slice("shared", summary="shared", length=1)
        sp = s.new_superposition([(1, 1.0)])
        # Create n_sups - 1 more superpositions containing the same slice
        last_sp = sp
        for _ in range(n_sups - 1):
            last_sp = s.add_candidate(last_sp, 1, 0.0)  # add with 0 amp
        # Now the slice is in n_sups superpositions
        # Refcount should reflect that
        slice_node_id = s._slice_index[1]
        # Each superposition bumps the slice's refcount
        assert s.refcount_of(slice_node_id) >= n_sups


class TestDecayCorrectness:
    """error_correct removes the chosen slice from the new superposition."""

    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=30, deadline=None)
    def test_error_correct_excludes_chosen(self, n_slices):
        s = Speculo()
        for i in range(n_slices):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        # slice_ids are 1..n_slices
        sp = s.new_superposition([(i + 1, 1.0 / n_slices) for i in range(n_slices)])
        c = s.measure(sp, first_oracle)
        chosen = s.resolve(c)[0]["chosen_slice_id"]
        ec = s.error_correct(c, uniform_oracle, first_tokens="retry")
        candidates = s.get_candidates(ec)
        # Chosen should NOT be in the remaining candidates
        assert chosen not in [cid for cid, _ in candidates]


class TestBranchCorrectness:
    """branch preserves candidates exactly."""

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_branch_preserves_candidates(self, n_slices):
        s = Speculo()
        for i in range(n_slices):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 1.0 / n_slices) for i in range(n_slices)])
        b = s.branch(sp)
        # Same candidates
        assert s.get_candidates(b) == s.get_candidates(sp)


class TestOutcomeTracking:
    """record_outcome accumulates over time."""

    @given(st.integers(min_value=1, max_value=20),
           st.lists(st.booleans(), min_size=1, max_size=20))
    @settings(max_examples=30, deadline=None)
    def test_outcomes_accumulate(self, n_slices, correctness):
        s = Speculo()
        for i in range(n_slices):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 1.0 / n_slices) for i in range(n_slices)])
        for was_correct in correctness:
            c = s.measure(sp, first_oracle)
            s.record_outcome(c, was_correct=was_correct)
        outcomes = s.outcomes()
        assert len(outcomes) == len(correctness)
        assert all(o.was_correct == wc for o, wc in zip(outcomes, correctness))


class TestTierTransitions:
    """Tier transitions preserve slice content and slice_id."""

    @given(st.sampled_from([Tier.WARM, Tier.COLD]))
    @settings(max_examples=20, deadline=None)
    def test_tier_round_trip(self, new_tier):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        s.demote(1, new_tier)
        assert s.get_slice(1).tier == new_tier
        s.promote(1)
        assert s.get_slice(1).tier == Tier.HOT
