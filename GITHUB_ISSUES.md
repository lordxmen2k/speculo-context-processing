# GitHub Issues — Speculo

Source: *Speculo: a persistent data structure for speculative context states with quantum semantics* (in preparation). Each issue below is referenced from the chapter. Issues are ordered by priority (P0 = must-do, P1 = high-value, P2 = nice-to-have, P3 = bookkeeping).

When pasting into GitHub, replace `[BOOK-CITE]` with the actual chapter reference and link to https://github.com/lordxmen2k/speculo-context-processing.

---

## P0-01: Lock-free Rust port — implement the production data structure

**Priority:** P0 (blocks the entire research claim)
**Labels:** `enhancement`, `rust`, `lock-free`, `good first issue` (for the scaffolding tasks)
**Estimated effort:** ~3 months

### Background

The book claims the Speculo data structure is the substrate for
lock-free speculative context processing, but the entire `rust/`
directory is scaffold-only. The Python reference uses a per-Speculo
mutation lock for thread safety, which is correct but not
lock-free. The actual research claim requires the Rust port with
per-node CAS and crossbeam-epoch.

### What needs to be done

1. Initialize the `rust/` crate
2. Define `SliceNode`, `SuperpositionNode`, `CollapsedNode` structs
3. Implement the `mapping` table as `DashMap<NodeID, Arc<Node>>`
4. Implement `add_slice`, `new_superposition`, `add_candidate`,
   `measure`, `error_correct`, `refresh`, `evict_stale`,
   `collapse_to`, `branch`
5. Replace the deferred-free list with `crossbeam-epoch::Guard::defer()`
6. Pass the same 71 tests as the Python reference (ported as `cargo test`)

### Acceptance criteria

- All 71 Python tests have Rust counterparts that pass
- The lock-free read path is verified: many readers don't block
  on the mutating path
- The Born-rule invariant (sum of amplitudes = 1.0) holds under
  concurrent load

### Book reference

Ch 5 §5.3, Ch 8 §8.2.

---

## P0-02: Real LLM integration — wire Speculo into an inference engine

**Priority:** P0
**Labels:** `enhancement`, `integration`
**Estimated effort:** 4-6 weeks

### Background

The Python reference is the data structure. The real value comes
from wiring it into an LLM inference engine (vLLM, SGLang, or
custom) so that the oracle gets called with the LLM's first K
tokens, and the system reverts on wrong answer.

### What needs to be done

1. Pick an inference engine (vLLM is the most likely)
2. Hook the model.generate() output stream
3. On each new token, call the oracle to update the superposition
4. On wrong answer (detected post-hoc), call error_correct
5. Add real benchmarks: end-to-end latency, accuracy, throughput

### Acceptance criteria

- A long-context task (e.g., 100K token document + question) runs
  with Speculo enabled
- End-to-end latency is faster than vanilla vLLM by 1.5-2×
- Accuracy on LongBench / RULER is unchanged

### Book reference

Ch 14 §14.2.

---

## P0-03: Embedding-based oracle with online learning

**Priority:** P0
**Labels:** `enhancement`, `oracle`
**Estimated effort:** 2-3 weeks

### Background

The current oracles are:
- `uniform_oracle` (baseline, no signal)
- `first_oracle` (always picks the first candidate, for testing)
- `keyword_oracle` (basic keyword overlap)
- `random_oracle` (random selection)
- `embedding_oracle` (placeholder, not production-ready)

We need a production oracle that:
1. Uses a sentence-transformers embedding to compare the query
   to each candidate's summary
2. Calibrates amplitudes from past outcomes (online learning)
3. Caches embeddings for speed

### What needs to be done

1. Implement a real `embedding_oracle` using sentence-transformers
2. Add online amplitude calibration: when `record_outcome` is
   called, refresh the relevant candidates' amplitudes
3. Cache embeddings to avoid recomputing
4. Add tests for accuracy on a synthetic workload

### Acceptance criteria

- The embedding oracle beats the keyword oracle on accuracy
  (e.g., 80%+ accuracy on a 100-candidate test set)
- Online calibration converges within 50 outcomes
- The oracle adds < 10ms per call for 100 candidates

### Book reference

Ch 6 §6.4, Ch 9 §9.3.

---

## P0-04: Born-rule amplitude sum verification as a runtime invariant

**Priority:** P0
**Labels:** `enhancement`, `correctness`
**Estimated effort:** 2-3 days

### Background

The Born rule (sum of amplitudes in a SuperpositionNode = 1.0) is
the core invariant. Currently it's checked only by tests. We
should make it a runtime invariant that fails loudly if violated.

### What needs to be done

1. Add a `_verify_born_rule` method that checks the sum
2. Call it after every mutating op
3. Add a `strict` mode that raises on violation vs a `lenient` mode
   that logs a warning
4. Add a property test that runs all ops many times and verifies
   the invariant holds

### Acceptance criteria

- A debug mode that catches invariant violations
- A release mode that's fast (no per-op check)
- Documented in the API

---

## P0-05: `merge_superpositions` for combining two speculative states

**Priority:** P0
**Labels:** `enhancement`, `data-structure`
**Estimated effort:** 1 week

### Background

When two agents are working on the same problem and each has its
own SuperpositionNode, we want to merge them into a single
SuperpositionNode. The amplitudes compose multiplicatively (or
additively, depending on the regime).

### What needs to be done

1. Add `merge_superpositions(sp_a, sp_b, mode="multiplicative") -> NodeID`
2. The new node has candidates that are in both, with amplitudes
   computed by the chosen mode
3. SliceID overlap is handled correctly
4. Test that the Born rule holds in the merged node

### Acceptance criteria

- `merge_superpositions(sp_a, sp_b)` creates a new node with
  candidates that are in both
- Sum of amplitudes = 1.0 after merge
- The original two are unchanged

---

## P0-06: LLM-based oracle with confidence thresholding

**Priority:** P0
**Labels:** `enhancement`, `oracle`
**Estimated effort:** 2 weeks

### Background

The embedding oracle is fast but limited. The LLM-based oracle
(small LLM evaluates each candidate) is more accurate but slow.
A hybrid: use the embedding oracle for fast filtering, then
LLM-based for high-confidence decisions.

### What needs to be done

1. Add an `llm_oracle` that uses a small LLM to evaluate candidates
2. Add a confidence threshold: if embedding oracle's confidence
   is above the threshold, use it; else fall back to LLM oracle
3. Cache LLM decisions to avoid re-evaluating
4. Test on a real workload

### Acceptance criteria

- The hybrid oracle is faster than pure LLM oracle
- Accuracy is close to pure LLM oracle (within 5%)

---

## P0-07: Persistent snapshot to disk

**Priority:** P0
**Labels:** `enhancement`, `persistence`
**Estimated effort:** 2 weeks

### Background

The Python reference is in-memory. To persist a Speculo instance
across process restarts, we need serialization.

### What needs to be done

1. Add `save_snapshot(root, path) -> None` that serializes the
   DAG (or just the path) to disk
2. Add `load_snapshot(path) -> Speculo, root` that deserializes
3. Use a compact format (msgpack, parquet, or custom binary)
4. Add tests for round-trip correctness

### Acceptance criteria

- A saved snapshot can be loaded in a fresh process and produce
  the same state
- The save format is documented and versioned

### Book reference

Ch 9 §9.4.

---

## P0-08: Bench against SpeContext / PRR on a real long-context task

**Priority:** P0
**Labels:** `benchmark`
**Estimated effort:** 3-4 weeks

### Background

The chapter needs a fair head-to-head benchmark of Speculo
against SpeContext, PRR, etc. The benchmark should measure:
end-to-end latency, accuracy, throughput, memory.

### What needs to be done

1. Set up the same long-context task in all systems
2. Run on the same model and the same context size
3. Measure all of the above
4. Write up the results

### Acceptance criteria

- All systems run the same workload
- The results table is reproducible
- The comparison is honest about what Speculo does and doesn't
  do better

### Book reference

Ch 15 §15.2.

---

## P0-09: The "speculative state DAG" used by multiple oracles

**Priority:** P0
**Labels:** `enhancement`, `multi-oracle`
**Estimated effort:** 1-2 weeks

### Background

Currently each SuperpositionNode has one set of amplitudes (one
oracle). For complex queries, multiple oracles may give different
amplitudes. We want a SuperpositionNode that holds multiple
amplitude vectors (one per oracle), and the measurement can
combine them.

### What needs to be done

1. Change `candidates` to be `dict[oracle_name, list[(slice_id, amplitude)]]`
2. The `measure` op takes an oracle name and uses the corresponding
   amplitudes
3. A new op `measure_combined(weights)` combines multiple oracles
4. Test that the Born rule holds for each oracle's vector

### Acceptance criteria

- A SuperpositionNode can hold amplitude vectors from multiple
  oracles
- The combined measure produces a sensible result

---

## P0-10: Cross-process sharing (mmap or shared memory)

**Priority:** P0
**Labels:** `enhancement`, `concurrency`
**Estimated effort:** 2-3 weeks

### Background

A long-running LLM might span multiple processes (the LLM
inference process, the tool execution process, the Speculo
process). They need to share the speculative state.

### What needs to be done

1. Investigate mmap or shared memory for the mapping table
2. Implement a process-safe Speculo that can be shared via
   mmap
3. Verify that multiple processes can read concurrently

### Acceptance criteria

- Two processes can read the same Speculo DAG via mmap
- Writes from one process are visible to the other
- Performance is within 2x of single-process Speculo

---

## P1-11: Tree of speculations (nested superpositions)

**Priority:** P1
**Labels:** `enhancement`, `data-structure`
**Estimated effort:** 1-2 weeks

### Background

A SuperpositionNode can be a candidate in a higher-level
SuperpositionNode. This gives a "tree of speculations" — a way
to model hierarchical or multi-level speculation.

### What needs to be done

1. Support SuperpositionNode as a candidate in new_superposition
2. When measuring, decide whether to collapse the outer or the
   inner first
3. Amplitudes compose multiplicatively
4. Test the tree traversal and measurement

### Acceptance criteria

- A SuperpositionNode can be a candidate in another SuperpositionNode
- Measuring the outer propagates correctly to the inner
- The Born rule holds in both levels

### Book reference

Ch 7 §7.3.

---

## P1-12: Real oracle implementations (learned, LLM-based, hybrid)

**Priority:** P1
**Labels:** `enhancement`, `oracle`
**Estimated effort:** 4-6 weeks

### Background

The current oracles are baselines. Real production oracles are:
- Learned: trained on past question/speculation pairs
- LLM-based: small LLM evaluates each candidate
- Hybrid: embedding-based for fast filter, LLM for high-conf decisions

### What needs to be done

1. Implement a learned oracle (train on data)
2. Implement an LLM-based oracle (call a small model)
3. Implement the hybrid oracle (P0-06)
4. Benchmark all three on a real workload
5. Publish the trained model

### Acceptance criteria

- The learned oracle is faster than the LLM oracle and more
  accurate than the embedding oracle
- The hybrid oracle is the best of both worlds

---

## P1-13: Speculo for multi-tenant serving

**Priority:** P1
**Labels:** `enhancement`, `multi-tenant`
**Estimated effort:** 2-3 weeks

### Background

In multi-tenant LLM serving, many requests share system prompts
and context. Speculo can dedupe the speculative state across
tenants using structural sharing.

### What needs to be done

1. Multi-tenant test setup (100 simulated tenants)
2. Measure sharing efficiency (refcount-based)
3. Compare to per-tenant Speculo instances
4. Document the savings

### Acceptance criteria

- 100 tenants sharing a common prefix use O(1) memory for the
  prefix (not O(100))

---

## P1-14: Integration with vLLM

**Priority:** P1
**Labels:** `enhancement`, `integration`
**Estimated effort:** 4-6 weeks

### Background

vLLM is the most popular open-source inference engine. A Speculo
plugin for vLLM would let vLLM users benefit from speculative
context processing without changing their API.

### What needs to be done

1. Implement a `SpeculoEngine` that wraps vLLM's `Engine`
2. Hook the model.generate() output stream
3. On each new token, call the oracle
4. On wrong answer, call error_correct
5. Add e2e tests with vLLM serving a long-context model

### Acceptance criteria

- vLLM with Speculo is faster than vanilla vLLM on long-context
- The Speculo engine passes vLLM's existing test suite
- The plugin is installable via `pip install speculo-vllm`

### Book reference

Ch 14 §14.3.

---

## P1-15: Visualization tool (DAG viewer)

**Priority:** P1
**Labels:** `enhancement`, `tooling`
**Estimated effort:** 1 week

### Background

A visual representation of the speculative state DAG would help
debugging. Something like Graphviz output or a simple HTML
viewer.

### What needs to be done

1. Add a `to_graphviz()` method that returns a DOT representation
2. Add a simple HTML viewer that renders the DAG in the browser
3. Show nodes colored by tier, edges labeled by oracle confidence

### Acceptance criteria

- `speculo.to_graphviz()` returns valid DOT
- A demo HTML page renders the DAG with hover-to-see-state

---

## P2-16: Performance optimization pass

**Priority:** P2
**Labels:** `performance`
**Estimated effort:** 1 week

### Background

After all the P0/P1 work is done, the Python reference will have
many small inefficiencies. A profiling pass would catch the
worst ones.

### What needs to be done

1. Profile the existing operations
2. Identify the top 5 hotspots
3. Optimize them (carefully, with tests)
4. Re-measure and report the speedup

### Acceptance criteria

- All 71 tests still pass
- At least 2x speedup on the most common operations

---

## P2-17: Compression (zstd or similar)

**Priority:** P2
**Labels:** `enhancement`, `storage`
**Estimated effort:** 3-5 days

### Background

A 10K-slice DAG with large slice payloads could be compressed
on disk. zstd or similar would give 2-5x reduction.

### What needs to be done

1. Add a `compress_slices` config option
2. Compress slice payloads on write
3. Decompress on read

### Acceptance criteria

- 10K-slice DAG with 1KB slices takes ~3MB on disk instead of ~10MB

---

## P2-18: Query language (filter the DAG by predicate)

**Priority:** P2
**Labels:** `enhancement`, `query`
**Estimated effort:** 1 week

### Background

A user might want to ask: "show me all slices that came from
the Java topic." A small query language would let them do this
without walking the DAG manually.

### What needs to be done

1. Add `query(predicate) -> list[SliceID]` that walks the DAG
   and returns matching slices
2. Predicate can be: tier, summary contains string, content matches regex
3. Add `query_summaries(text) -> list[SliceID]` shortcut

### Acceptance criteria

- `query_summaries('Python')` returns all slices with "Python"
  in the summary

---

## P2-19: SQLite backend for persistence

**Priority:** P2
**Labels:** `enhancement`, `storage`
**Estimated effort:** 1-2 weeks

### Background

For long-running agents that accumulate many slices, an SQLite
backend for persistence would let the Speculo state scale
beyond RAM.

### What needs to be done

1. Implement `SQLiteBackend` that uses SQLite for COLD slices
2. Add a config option `cold_backend='sqlite'`
3. Test with a DAG that's larger than RAM

### Acceptance criteria

- A DAG with 100K slices (most COLD) uses < 1GB RAM
- COLD reads are < 50ms

---

## P3-20: Documentation site (mkdocs or similar)

**Priority:** P3
**Labels:** `docs`
**Estimated effort:** 3-5 days

### Background

A docs site (mkdocs, sphinx, docusaurus) would make the API
discoverable.

### What needs to be done

1. Set up mkdocs with the `material` theme
2. Add API reference (auto-generated from docstrings)
3. Add tutorials for common operations
4. Add the design doc as a chapter

### Acceptance criteria

- `mkdocs serve` works locally
- The site includes API reference, tutorials, and the design

---

## P3-21: CI/CD (GitHub Actions)

**Priority:** P3
**Labels:** `ci`
**Estimated effort:** 2-3 days

### Background

Currently no CI. Tests are run manually.

### What needs to be done

1. Add `.github/workflows/test.yml`
2. Run on every push and PR
3. Run on Python 3.10, 3.11, 3.12, 3.13, 3.14

### Acceptance criteria

- CI runs on every push
- All 71 tests pass on CI
- The CI badge is in the README

---

## P3-22: Type stubs (.pyi) for editor support

**Priority:** P3
**Labels:** `tooling`
**Estimated effort:** 2-3 days

### Background

The Python reference has type hints but no `.pyi` stubs.
Editors can use stubs for better autocomplete and type
checking.

### What needs to be done

1. Generate or write `.pyi` files for the public API
2. Verify mypy/pyright is happy

### Acceptance criteria

- A user gets autocomplete on `Speculo.add_slice` in their editor
- mypy --strict passes on the stubs

---

## P3-23: Test coverage push to 90%+

**Priority:** P3
**Labels:** `tests`
**Estimated effort:** 1 day

### Background

Coverage report shows ~80% on the Python reference. The
uncovered paths are mostly error-handling code.

### What needs to be done

1. Run `pytest --cov=src/speculo` and identify uncovered lines
2. Add tests for the uncovered paths
3. Aim for 90%+ coverage

### Acceptance criteria

- Coverage is 90%+
- All uncovered lines have a corresponding test
