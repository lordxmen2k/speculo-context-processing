"""Microbenchmarks for Speculo.

Per-op throughput on a single thread.
Compares add_slice, new_superposition, measure, error_correct,
refresh, branch.

Run: python -m bench.bench_speculo
"""
import sys
import time
sys.path.insert(0, "src")

from speculo import Speculo, uniform_oracle, first_oracle, keyword_oracle


def time_it(name: str, n: int, op) -> float:
    start = time.perf_counter()
    for _ in range(n):
        op()
    elapsed = time.perf_counter() - start
    ops = n / elapsed
    print(f"  {name:48s} {ops:>10,.0f} ops/sec  ({elapsed*1000:.1f}ms for {n})")
    return ops


def main():
    print("Speculo microbenchmarks (single-threaded)")
    print("=" * 60)

    # Setup: 10 slices
    s = Speculo()
    slice_ids = []
    for i in range(10):
        sid = s.add_slice(f"slice_{i}", summary=f"summary_{i}", length=10)
        slice_ids.append(sid)
    sp = s.new_superposition([(sid, 0.1) for sid in slice_ids])
    print(f"\nSetup: 10 slices + 1 superposition of 10 candidates")
    print(f"  stats: {s.stats()}")
    print()

    print("[1] Mutation throughput")
    n = 1000
    s_setup = Speculo()
    counter = [0]
    def add_slice_op():
        counter[0] += 1
        s_setup.add_slice(f"x{counter[0]}", summary="x", length=1)
    time_it("add_slice x 1000", n, add_slice_op)

    s_setup2 = Speculo()
    sids = [s_setup2.add_slice(f"x{i}", summary=f"x{i}", length=1) for i in range(5)]
    def new_sup_op():
        s_setup2.new_superposition([(sid, 0.2) for sid in sids])
    time_it("new_superposition(5 candidates) x 1000", n, new_sup_op)

    counter = [0]
    def add_candidate_op():
        counter[0] += 1
        new_sid = s_setup2.add_slice(f"y{counter[0]}", summary="y", length=1)
        s_setup2.add_candidate(sp, new_sid, 0.05)
    time_it("add_candidate x 1000", n, add_candidate_op)

    def measure_op():
        c = s.measure(sp, first_oracle)
        # Reset
        s.error_correct(c, uniform_oracle, first_tokens="r")
    time_it("measure + error_correct cycle x 1000", n // 2, measure_op)

    def refresh_op():
        s.refresh(sp, slice_ids[0], 0.2)
    time_it("refresh x 1000", n, refresh_op)

    def branch_op():
        s.branch(sp)
    time_it("branch x 1000", n, branch_op)

    print()
    print("[2] Read throughput")
    n = 10_000
    time_it("get_candidates x 10000", n,
            lambda: s.get_candidates(sp))
    time_it("amplitude_sum x 10000", n,
            lambda: s.amplitude_sum(sp))
    time_it("is_superposition x 10000", n,
            lambda: s.is_superposition(sp))
    time_it("stats x 1000", 1_000,
            lambda: s.stats())
    time_it("resolve(sp) x 1000", 1_000,
            lambda: s.resolve(sp))

    print()
    print("[3] Stress: many candidates per superposition")
    for n_cands in [10, 50, 100]:
        s_setup3 = Speculo()
        sids = [s_setup3.add_slice(f"x{i}", summary=f"x{i}", length=1) for i in range(n_cands)]
        sp3 = s_setup3.new_superposition([(sid, 1.0 / n_cands) for sid in sids])
        n_iters = 100
        def measure_many():
            c = s_setup3.measure(sp3, first_oracle)
            s_setup3.error_correct(c, uniform_oracle, first_tokens="r")
        time_it(f"measure cycle on {n_cands}-candidate superposition x {n_iters}",
                n_iters, measure_many)

    print()
    print("Note: single-threaded with a global mutation lock.")
    print("Concurrent throughput is measured in test_speculo_concurrent.py.")


if __name__ == "__main__":
    main()
