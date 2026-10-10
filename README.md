# Speculo — speculative context processing with quantum semantics

A persistent data structure for managing **speculative context
states** when serving long-context LLMs. The system holds multiple
candidate context slices in *superposition*, then *collapses* to
one when the LLM produces output. Wrong collapses fall back
gracefully; the persistent DAG keeps every state recoverable.

**Status:** design + Python reference. The data structure is
specified in `tech_note.md`. The Python reference is the
correctness oracle; a real serving-system integration would
target vLLM, SGLang, or a custom inference engine.

## The pitch

Long-context LLMs (1M+ tokens) are bottlenecked by context
processing, not generation. Every existing solution tries to
either:
- Compress the context (lossy, breaks faithfulness)
- Speculate about a subset (SpeContext, PRR, LongSpec — all
  tied to one attention mechanism)
- Use parallel prompt paths (Superposition Prompting — no
  persistent data structure)

Speculo is the **persistent data structure layer** that those
algorithms need. A superposition node holds N candidate context
slices with amplitudes (probabilities). An oracle function
collapses to one slice when the LLM begins generating. If the
collapse is wrong, the system re-enters superposition with the
remaining candidates. Every state is persistent — you can
snapshot, rewind, or branch to try a different collapse order.

## The quantum analogy

We take the quantum mechanics metaphor seriously as a **data
structure design tool**, not as a runtime trick:

| quantum construct | speculative context analog |
|---|---|
| superposition | multiple candidate context slices exist simultaneously |
| measurement / collapse | LLM's first output token selects one slice |
| Born rule | amplitude ≈ P(this is the right slice) |
| amplitude amplification (Grover) | quadratic speedup on finding the right slice |
| decoherence | speculation results become stale; need to refresh |
| quantum error correction | detect wrong collapse; fall back to next-best |

The contribution is the **data structure** that supports all of
these as first-class operations, with persistent semantics so
you can rewind, snapshot, and branch.

## What ships in v0

- [x] `tech_note.md` — design proposal (v0.1)
- [x] `Speculo` class — persistent DAG of speculative states
- [x] superposition, oracle, collapse, decoherence, error correction ops
- [x] 50+ unit tests + 9+ property tests + concurrent stress tests
- [x] `bench/REPORT.md` and `COMPARISON.md` (vs SpeContext, PRR, etc.)
- [ ] Real LLM integration (Phase 2 — needs vLLM/SGLang on Linux+CUDA)

## Project layout

```
Speculo-Context-Processing/
  .gitignore
  LICENSE                                Apache 2.0
  README.md                              this file
  tech_note.md                           design proposal
  GITHUB_ISSUES.md                       prioritized issues P0-P3
  pyproject.toml
  NUMBERS.md                             quick-reference perf numbers
  COMPARISON.md                          vs SpeContext, PRR, etc.
  src/speculo/
    __init__.py                          public API
    types.py                             NodeID, Slice, Amplitude
    core.py                              Speculo (the persistent DAG)
    oracle.py                            Oracle (the collapse function)
  tests/
    test_speculo_basic.py
    test_speculo_property.py
    test_speculo_concurrent.py
    test_speculo_quantum.py              Born rule, decoherence, etc.
  examples/
    long_context_demo.py                 long-context LLM + Speculo
  bench/
    bench_speculo.py
    bench_long_context.py
    REPORT.md
```

## License

Apache 2.0. See [`LICENSE`](./LICENSE).

## Author

Gerald Enrique Nelson Mc Kenzie — lordxmen2k@gmail.com
