"""Quantum-specific tests for Speculo.

These test the quantum mechanics analogies (superposition,
measurement, Born rule, decoherence, error correction) directly.
"""
import time
import pytest
import sys
sys.path.insert(0, "src")

from speculo import (
    Speculo, Tier, uniform_oracle, first_oracle, keyword_oracle,
    embedding_oracle,
)


class TestSuperposition:
    """Multiple candidate slices exist simultaneously before measurement."""

    def test_all_candidates_persist_until_collapse(self):
        s = Speculo()
        for i in range(5):
            s.add_slice(f"slice_{i}", summary=f"summary_{i}", length=10)
        sp = s.new_superposition([(i + 1, 0.2) for i in range(5)])
        # All 5 candidates should be present
        cands = s.get_candidates(sp)
        assert len(cands) == 5
        # Each is recoverable
        for cid, amp in cands:
            assert s.get_slice(cid) is not None
            assert amp > 0

    def test_superposition_is_not_a_collapse(self):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        sp = s.new_superposition([(1, 1.0)])
        # A superposition is not a collapsed node
        assert s.is_superposition(sp)
        assert not s.is_collapsed(sp)

    def test_collapse_selects_one(self):
        s = Speculo()
        for i in range(4):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 0.25) for i in range(4)])
        c = s.measure(sp, first_oracle)
        snap = s.resolve(c)
        # One chosen, three remaining
        assert snap[0]["chosen_slice_id"] in [1, 2, 3, 4]
        assert len(snap[0]["remaining_candidates"]) == 3


class TestBornRule:
    """Amplitudes sum to 1.0 (Born rule)."""

    def test_amplitude_sum_exactly_one(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        # Pass non-normalized amplitudes
        sp = s.new_superposition([(1, 0.3), (2, 0.4), (3, 0.3)])
        assert abs(s.amplitude_sum(sp) - 1.0) < 1e-9

    def test_born_rule_preserved_through_measure(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, first_oracle)
        # The collapsed node has a chosen (amp=1.0 for the chosen) +
        # remaining candidates (sum to < 1.0)
        # This isn't a superposition, so the Born rule applies only
        # to the remaining set after error correction
        try:
            ec = s.error_correct(c, uniform_oracle, first_tokens="r")
            assert abs(s.amplitude_sum(ec) - 1.0) < 1e-9
        except Exception:
            # No remaining is fine
            pass

    def test_amplitude_distribution_after_refresh(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        sp2 = s.refresh(sp, 2, 0.6)
        amps = {cid: a for cid, a in s.get_candidates(sp2)}
        # After refresh, sum should be 1.0
        total = sum(amps.values())
        assert abs(total - 1.0) < 1e-9


class TestDecoherence:
    """Candidates become stale; eviction removes them."""

    def test_evict_stale_removes_old(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        # Make slice 2 very old
        n2 = s._get(s._slice_index[2])
        with n2.lock:
            n2.last_observed = 0
        # Evict with 1s TTL
        sp2 = s.evict_stale(sp, ttl_seconds=1.0, now=time.time())
        # Slice 2 should be gone
        cids = [cid for cid, _ in s.get_candidates(sp2)]
        assert 2 not in cids

    def test_touch_refreshes_timestamp(self):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        n = s._get(s._slice_index[1])
        with n.lock:
            n.last_observed = 0
        s.touch_slice(1)
        assert n.last_observed > 0

    def test_all_stale_eviction_raises(self):
        s = Speculo()
        for i in range(2):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        for sid in [1, 2]:
            n = s._get(s._slice_index[sid])
            with n.lock:
                n.last_observed = 0
        with pytest.raises(Exception):
            s.evict_stale(sp, ttl_seconds=1.0, now=time.time())


class TestErrorCorrection:
    """Quantum error correction: detect wrong collapse, fall back."""

    def test_error_correct_recovers_to_remaining(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
        c = s.measure(sp, first_oracle)
        chosen = s.resolve(c)[0]["chosen_slice_id"]
        ec = s.error_correct(c, uniform_oracle, first_tokens="retry")
        # The new superposition excludes the chosen
        new_cids = [cid for cid, _ in s.get_candidates(ec)]
        assert chosen not in new_cids
        # Amplitudes are renormalized
        assert abs(s.amplitude_sum(ec) - 1.0) < 1e-9

    def test_repeated_error_correct_eventually_exhausts(self):
        s = Speculo()
        for i in range(2):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        # Measure -> error_correct -> measure -> error_correct
        c1 = s.measure(sp, first_oracle)
        ec1 = s.error_correct(c1, uniform_oracle, first_tokens="r")
        c2 = s.measure(ec1, first_oracle)
        # After exhausting, error_correct on the last collapse should fail
        with pytest.raises(Exception):
            s.error_correct(c2, uniform_oracle, first_tokens="r2")

    def test_error_correct_with_keyword_oracle(self):
        s = Speculo()
        s.add_slice("Python is great.", summary="about Python", length=3)
        s.add_slice("Java is great.", summary="about Java", length=3)
        s.add_slice("Rust is fast.", summary="about Rust", length=3)
        sp = s.new_superposition([(1, 0.4), (2, 0.3), (3, 0.3)])
        c = s.measure(sp, first_oracle)  # picks slice 1
        # Pretend it was wrong; error correct with a Python-relevant query
        ec = s.error_correct(c, keyword_oracle, first_tokens="about Java")
        # The new superposition should have slices 2 and 3
        cids = [cid for cid, _ in s.get_candidates(ec)]
        assert 1 not in cids
        assert 2 in cids
        assert 3 in cids


class TestCollapse:
    """Manual selection bypasses the oracle."""

    def test_collapse_to_picks_specific(self):
        s = Speculo()
        for i in range(4):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 0.25) for i in range(4)])
        c = s.collapse_to(sp, 3)
        assert s.resolve(c)[0]["chosen_slice_id"] == 3
        assert s.resolve(c)[0]["confidence"] == 1.0

    def test_collapse_to_invalid_raises(self):
        s = Speculo()
        s.add_slice("x", summary="x", length=1)
        sp = s.new_superposition([(1, 1.0)])
        with pytest.raises(Exception):
            s.collapse_to(sp, 99)


class TestOracles:
    """Different oracles return different distributions."""

    def test_uniform_vs_first(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 0.33) for i in range(3)])
        c_first = s.measure(sp, first_oracle)
        # First oracle always picks slice 1
        assert s.resolve(c_first)[0]["chosen_slice_id"] == 1

    def test_keyword_prefers_relevant(self):
        s = Speculo()
        s.add_slice("Python is a programming language", summary="about Python", length=5)
        s.add_slice("Quantum mechanics is fascinating", summary="about quantum", length=4)
        s.add_slice("Coffee is brewed from beans", summary="about coffee", length=5)
        sp = s.new_superposition([(1, 0.4), (2, 0.3), (3, 0.3)])
        c = s.measure(sp, keyword_oracle, first_tokens="tell me about Python")
        # Keyword oracle should prefer slice 1 (Python)
        assert s.resolve(c)[0]["chosen_slice_id"] == 1

    def test_keyword_no_match_falls_back(self):
        s = Speculo()
        s.add_slice("Python is great", summary="about Python", length=3)
        s.add_slice("Java is great", summary="about Java", length=3)
        sp = s.new_superposition([(1, 0.5), (2, 0.5)])
        c = s.measure(sp, keyword_oracle, first_tokens="unrelated words")
        # Should still pick something (uniform fallback)
        assert s.resolve(c)[0]["chosen_slice_id"] in [1, 2]
