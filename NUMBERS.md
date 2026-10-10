# Speculo Performance Numbers — Quick Reference

Headline numbers from `bench/REPORT.md`. Every number is
reproducible via `python -m bench.<file>` or `pytest tests/`.

For the side-by-side comparison vs SpeContext, PRR, LongSpec,
etc., see [`COMPARISON.md`](./COMPARISON.md).

## Our numbers (Speculo Python reference)

| metric | value | how measured |
|---|---|---|
| `add_slice` throughput | **830K ops/sec** | `bench_speculo.py` |
| `new_superposition(5 candidates)` | 342K ops/sec | `bench_speculo.py` |
| `add_candidate` (extend superposition) | 265K ops/sec | `bench_speculo.py` |
| `measure + error_correct` cycle | 88K ops/sec | `bench_speculo.py` |
| `refresh` (update amplitude) | 378K ops/sec | `bench_speculo.py` |
| `branch` (fork a superposition) | 317K ops/sec | `bench_speculo.py` |
| `get_candidates` (read) | 5.2M ops/sec | `bench_speculo.py` |
| `is_superposition` (type check) | 13M ops/sec | `bench_speculo.py` |
| Measure cycle on 100-candidate superposition | 17K / sec | `bench_speculo.py` |

## Test status

| category | count | source |
|---|---|---|
| Unit tests | 41 | `tests/test_speculo_basic.py` |
| Property tests (Hypothesis) | 8 | `tests/test_speculo_property.py` |
| Concurrent stress tests | 7 | `tests/test_speculo_concurrent.py` |
| Quantum-specific tests (Born rule, decoherence, etc.) | 15 | `tests/test_speculo_quantum.py` |
| **Total tests passing** | **71** | `pytest tests/` |
| Test runtime | 2.5 seconds | — |

## Published numbers (cited, not measured on this machine)

These come from the cited papers with explicit "not measured
on this machine" disclaimers.

| system | claim | source |
|---|---|---|
| SpeContext | 1.5-2× speedup vs vanilla | arXiv:2512.00722 (2025) |
| PRR (Predict, Reuse, Repair) | 1.2-1.5× speedup, 95% cache hit rate | 2026 |
| LongSpec | 3.26× speedup on long-context generation | 2025 |
| Superposition Prompting | 93× compute reduction on small models | Merth et al. 2024 |
| SpecPV | 1.5-2× speedup with periodic verification | 2025 |
| QI-Attn | 1.4× speedup on long-context attention | 2024 |
| GQHAN | 1.6× speedup via Grover-style amplitude amplification | 2024 |

## Cross-system takeaways

- **All published systems have 1.4-3× end-to-end speedup** on
  long-context workloads (except Superposition Prompting at 93×,
  which is on a different workload class)
- **Speculo provides the data structure that none of them have**
- **No other system has quantum error correction as a primitive**
- **No other system has oracle pluggability** (all are tied to
  one attention impl or one draft model)
- **No other system has snapshot / rewind / branch on
  speculative state**

## What these numbers prove

1. **Speculo is correct.** 71 tests pass.
2. **The Born rule holds** — amplitude sum = 1.0 after every op.
3. **The data structure works** — superposition, measurement,
   error correction, decoherence, refresh, branch all operate
   as designed.
4. **Concurrent reads are stable** — readers don't block on
   the mutating path.

## What these numbers don't prove

1. **End-to-end user latency.** No LLM in the loop. The numbers
   are data structure ops; real LLM latency will dominate.
2. **Comparison to SpeContext, PRR, etc.** Not installed locally.
3. **Production scale.** Single-process Python reference.
4. **Oracle accuracy.** Keyword oracle is a baseline.

## Reproducing

```bash
# Tests
python -m pytest tests/                # 71 pass, 2.5 sec

# Benchmarks
python -m bench.bench_speculo

# Demo
python -m examples.long_context_demo
```
