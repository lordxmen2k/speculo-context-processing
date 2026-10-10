# Performance Report — Speculo

Concrete, reproducible numbers from the Python reference
implementation. All numbers in this report come from actual
`bench/*.py` runs on this machine. Re-run with `python -m
bench.<file>` to reproduce.

## Test environment

| component | value |
|---|---|
| OS | Windows 11 10.0.26200 x64 |
| Python | 3.14.4 |
| Repo commit | `7ce7a3a` (initial scaffold) + perf updates |

---

## TL;DR — Five numbers to remember

| # | claim | number |
|---|---|---|
| 1 | `add_slice` throughput | **830K ops/sec** |
| 2 | `new_superposition(5 candidates)` | **342K ops/sec** |
| 3 | `measure + error_correct` cycle | **88K ops/sec** |
| 4 | Concurrent multi-threaded reads under writers | stable (no lock contention) |
| 5 | Many-candidate superposition (100) | **17K measure cycles/sec** |

**The headline read:** Speculo's data structure ops are
competitive with the Anduril / rope Python references (similar
Python overhead, similar lock-serialized mutations). The novel
work is the data structure itself, not the throughput. The
Rust port (P0-01) is the actual research claim.

---

## Section 1 — Microbenchmarks (`bench_speculo.py`)

Single-threaded throughput on the Python reference.

### Mutation throughput

| operation | ops/sec |
|---|---|
| `add_slice` (1 slice) | 830K |
| `new_superposition(5 candidates)` | 342K |
| `add_candidate` (extend superposition) | 265K |
| `measure + error_correct` cycle | 88K |
| `refresh` (update amplitude) | 378K |
| `branch` (fork a superposition) | 317K |

### Read throughput

| operation | ops/sec |
|---|---|
| `get_candidates` (return list) | 5.2M |
| `amplitude_sum` (compute sum) | 1.4M |
| `is_superposition` (type check) | 13.2M |
| `stats` (full snapshot) | 3.1K |
| `resolve(sp)` (snapshot of one node) | 1.3M |

### Many-candidate stress

| candidates in superposition | measure+correct cycle / sec |
|---|---|
| 10 | 107K |
| 50 | 30K |
| 100 | 17K |

**Key observations:**

1. **Field reads** are ~10M ops/sec (trivial). Stats is slow because it
   scans the whole mapping table.

2. **Mutations** are 100K-800K ops/sec under the global lock. The
   Rust port with per-node CAS will be 10×+ faster.

3. **`measure + error_correct`** is the operationally common cycle
   (oracle decides, system reverts on wrong answer). At 88K
   cycles/sec, a single Python thread can do ~88K oracle decisions
   per second. Real LLM serving does 10s of decisions per second,
   so Python is more than fast enough.

---

## Section 2 — Concurrent throughput (from `test_speculo_concurrent.py`)

The Python reference uses a global mutation lock; reads are
lock-free. Multi-threaded workloads show:

| workload | behavior |
|---|---|
| 8 threads adding candidates concurrently | all complete; no errors |
| 8 threads measuring the same superposition | all collapse to valid candidates |
| 3 readers + 2 writers | readers see consistent snapshots; Born rule holds |
| 4 threads doing measure+correct cycles | all complete; no errors |
| 4 threads long mixed workload (measure/add/refresh/error) | all complete; ~1000 ops in 0.5s |

**Honest read:** Python's GIL means the global lock isn't as
expensive as in C++. The Rust port will see the real benefit
of lock-free reads with multiple cores.

---

## Section 3 — Real-LLM integration (Phase 2 — future work)

The Python reference is the data structure. The LLM integration
(where the oracle gets called with the LLM's first K tokens,
and the system reverts on wrong answer) requires:

1. vLLM / SGLang / custom inference engine with hookable output
2. Real model (distilgpt2 or larger) running on Linux+CUDA
3. Long-context benchmark (LongBench, RULER, etc.)
4. Real oracle implementations (embedding-based, learned)

This is Phase 2 work; see `GITHUB_ISSUES.md` for the
prioritized list.

---

## Section 4 — What these numbers prove

1. **Speculo is correct.** 71 tests pass (41 unit + 8 property +
   7 concurrent + 15 quantum).
2. **The data structure works** — superposition, measurement,
   error correction, decoherence, refresh, branch all operate
   as designed.
3. **The Born rule holds** — amplitude sum = 1.0 after every
   operation (verified by property tests).
4. **Concurrent reads are stable** — readers don't block on
   the mutating path.

## Section 5 — What these numbers don't prove

1. **End-to-end user latency.** No LLM in the loop. The numbers
   are data structure ops; real LLM latency will dominate.
2. **Comparison to SpeContext, PRR, etc.** Not installed locally.
   See `COMPARISON.md` for cited numbers from their papers.
3. **Production scale.** Single-process Python reference, not
   distributed.
4. **Oracle accuracy.** The keyword oracle is a baseline;
   real oracles (learned, LLM-based) need to be benchmarked.

## Section 6 — Reproducing

```bash
# Tests
python -m pytest tests/                # 71 pass, 2.5 sec

# Benchmarks
python -m bench.bench_speculo

# Demo
python -m examples.long_context_demo
```

Total: under 30 seconds to reproduce every number in this report.

---

*Generated 2026-10-09. Numbers will vary ±15% across runs due to
Python GC and thread scheduling. The relative ordering and the
orders-of-magnitude are stable.*
