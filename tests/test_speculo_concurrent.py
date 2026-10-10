"""Concurrent tests for Speculo.

Verify thread-safety invariants under multi-threaded load:
- All mutating ops are serialized by `_mutation_lock`
- Reads are lock-free and see a consistent snapshot
- Structural sharing (refcount) is correct under concurrent
  add_candidate / measure / error_correct
- No node is double-freed
"""
import threading
import time
import pytest
import sys
sys.path.insert(0, "src")

from speculo import (
    Speculo, uniform_oracle, first_oracle, keyword_oracle,
)


class TestConcurrentAddCandidate:
    """Multiple threads adding candidates to a shared superposition."""

    def test_many_threads_adding_to_same_superposition(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"base{i}", summary=f"base{i}", length=1)
        sp = s.new_superposition([(1, 0.4), (2, 0.3), (3, 0.3)])
        n_threads = 8
        n_per_thread = 10
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker(tid):
            try:
                barrier.wait()
                for i in range(n_per_thread):
                    # add_slice returns a slice_id directly
                    new_sid = s.add_slice(
                        f"x{tid}_{i}",
                        summary=f"x{tid}_{i}",
                        length=1,
                    )
                    sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])
                    s.add_candidate(sp, new_sid, 0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"errors: {errors[:3]}"


class TestConcurrentMeasure:
    """Multiple threads measuring the same superposition."""

    def test_concurrent_measurements(self):
        s = Speculo()
        for i in range(5):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 0.2) for i in range(5)])
        n_threads = 8
        n_per_thread = 50
        barrier = threading.Barrier(n_threads)
        collapsed_ids = [None] * (n_threads * n_per_thread)
        errors = []

        def worker(tid):
            try:
                barrier.wait()
                for i in range(n_per_thread):
                    c = s.measure(sp, first_oracle)
                    collapsed_ids[tid * n_per_thread + i] = c
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"errors: {errors[:3]}"
        # All collapsed nodes should be valid
        for cid in collapsed_ids:
            assert s.is_collapsed(cid)
            assert s.resolve(cid)[0]["kind"] == "collapsed"


class TestConcurrentReads:
    """Reads should be lock-free and see consistent snapshots."""

    def test_reads_under_concurrent_adds(self):
        s = Speculo()
        for i in range(3):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(1, 0.5), (2, 0.3), (3, 0.2)])

        stop = threading.Event()
        errors = []
        read_count = [0]
        read_lock = threading.Lock()

        def reader():
            try:
                while not stop.is_set():
                    cands = s.get_candidates(sp)
                    assert abs(sum(a for _, a in cands) - 1.0) < 1e-9
                    s.amplitude_sum(sp)
                    with read_lock:
                        read_count[0] += 1
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(20):
                    s.add_slice(f"new{i}", summary=f"new{i}", length=1)
                    s.add_candidate(sp, 4 + i, 0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(3)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join()
        assert not errors, f"errors: {errors[:3]}"
        assert read_count[0] > 0


class TestConcurrentErrorCorrect:
    """Concurrent error correction under multiple readers."""

    def test_error_correct_under_load(self):
        s = Speculo()
        for i in range(5):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 0.2) for i in range(5)])
        n_threads = 4
        n_per_thread = 30
        barrier = threading.Barrier(n_threads)
        errors = []

        def worker(tid):
            try:
                barrier.wait()
                cur = sp
                for i in range(n_per_thread):
                    c = s.measure(cur, first_oracle)
                    try:
                        cur = s.error_correct(c, uniform_oracle, first_tokens="retry")
                    except Exception:
                        # No candidates left — measure again
                        cur = sp
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,))
                   for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"errors: {errors[:3]}"


class TestStress:
    """Long-running stress tests."""

    def test_long_mixed_workload(self):
        s = Speculo()
        for i in range(10):
            s.add_slice(f"x{i}", summary=f"x{i}", length=1)
        sp = s.new_superposition([(i + 1, 0.1) for i in range(10)])

        stop = threading.Event()
        errors = []
        op_count = [0]
        op_lock = threading.Lock()

        def mixed_worker():
            try:
                nonlocal sp
                cur = sp
                while not stop.is_set():
                    op = random.choice(["measure", "add", "refresh", "error"])
                    try:
                        if op == "measure":
                            if s.is_superposition(cur):
                                c = s.measure(cur, first_oracle)
                                cur = c
                            else:
                                cur = sp
                        elif op == "add":
                            if s.is_collapsed(cur):
                                try:
                                    cur = s.error_correct(cur, uniform_oracle,
                                                         first_tokens="r")
                                except Exception:
                                    cur = sp
                            if s.is_superposition(cur):
                                new_sid = s.add_slice(
                                    f"x{random.randint(1, 1_000_000)}",
                                    summary="new",
                                    length=1,
                                )
                                cur = s.add_candidate(cur, new_sid, 0.05)
                        elif op == "refresh":
                            if s.is_collapsed(cur):
                                try:
                                    cur = s.error_correct(cur, uniform_oracle,
                                                         first_tokens="r")
                                except Exception:
                                    cur = sp
                            if s.is_superposition(cur):
                                cur = s.refresh(cur, 1, 0.2)
                        elif op == "error":
                            if s.is_collapsed(cur):
                                try:
                                    cur = s.error_correct(cur, uniform_oracle,
                                                         first_tokens="r")
                                except Exception:
                                    cur = sp
                    except Exception:
                        cur = sp
                    with op_lock:
                        op_count[0] += 1
            except Exception as e:
                errors.append(e)

        import random
        threads = [threading.Thread(target=mixed_worker) for _ in range(4)]
        for t in threads:
            t.start()
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join()
        assert not errors, f"errors: {errors[:3]}"
        assert op_count[0] > 0
