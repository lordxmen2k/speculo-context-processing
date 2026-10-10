# Speculo: a persistent data structure for speculative context states — design note (draft v0.1)

**Author:** Gerald Enrique Nelson Mc Kenzie
**Date:** 2026-10-09
**Status:** draft v0.1
**Target venue:** technical paper / chapter

## 1. Motivation

Long-context LLMs (1M+ tokens) are bottlenecked by context
processing, not generation. Every existing solution tries one
of three approaches:

- **Compress the context** (lossy, breaks faithfulness)
- **Speculate about a subset** (SpeContext, PRR, LongSpec — all
  tied to one attention mechanism or one speculation trigger)
- **Use parallel prompt paths** (Superposition Prompting — no
  persistent data structure)

None of them has a **persistent data structure** for managing
speculative context states with proper collapse semantics,
decoherence handling, and graceful fallback. That's the gap.

We argue that a quantum-inspired data structure — superposition
nodes, oracle-based collapse, Born-rule amplitude calibration,
decoherence timers, error-correction logs — is the right
abstraction. It composes with any of the existing algorithms
and gives the speculative state persistence, recoverability,
and a clean semantics.

## 2. The data structure

A persistent DAG of speculative context states. The node types
are:

- **SliceNode** — a concrete context slice (a sequence of tokens
  or KV blocks). Has a content hash, a length, and a tier
  (HOT/WARM/COLD, like Anduril).
- **SuperpositionNode** — holds N candidate SliceNodes, each
  with an amplitude (probability). Sum of amplitudes = 1.0
  (Born rule). The superposition is itself a SliceNode
  (composes: a superposition can be a candidate in a higher-
  level superposition).
- **CollapsedNode** — the result of a measurement. Points to
  the chosen SliceNode + a confidence score. Children are the
  remaining candidates (for error correction).

Operations:

- `add_candidate(slice, amplitude) → new_superposition` — extend
  a superposition with a new candidate. Amplitudes are
  renormalized.
- `measure(superposition, oracle) → collapsed` — apply the
  oracle function to choose one candidate. Returns a
  CollapsedNode.
- `error_correct(collapsed, oracle) → new_superposition` — the
  collapse was wrong; re-enter superposition with the remaining
  candidates (excluding the chosen one).
- `refresh(superposition, candidate) → new_superposition` —
  update the amplitude of a candidate (e.g., from new
  evidence).
- `evict_stale(superposition, ttl) → new_superposition` —
  remove candidates whose last-access time exceeds the TTL.
- `collapse_to(slice) → collapsed` — manually pick a candidate
  (bypass the oracle).
- `branch(superposition) → new_superposition` — fork the
  speculation: try a different oracle or different candidate
  set.

Invariants:

- **Sum of amplitudes in a SuperpositionNode = 1.0** (Born rule)
- **SliceNode content is immutable** (persistent roots)
- **CollapsedNode preserves the unchosen candidates as children**
  (so error correction can re-enter superposition)
- **Stable NodeIDs** (Bw-tree invariant — no reuse)
- **Structural sharing** across superpositions (same slice as
  a candidate in multiple superpositions, refcount on the
  SliceNode)

## 3. The oracle

The oracle is a function `(first_k_tokens, candidate_summaries)
→ probability_distribution`. The simplest impl: a small
embedding model that compares the question's embedding to
each candidate's summary embedding, and outputs a softmax.

A better impl: a learned model trained on past
question/speculation pairs (the data structure records every
collapse and its outcome, so this trains itself over time).

A production impl: a small LLM that gets the question + each
candidate's first 100 tokens and outputs "P(this is the right
slice)". Slower but more accurate.

The oracle is a pluggable component. The data structure
doesn't care which one you use.

## 4. Decoherence and TTL

Each SuperpositionNode has a `created_at` timestamp. Each
candidate has a `last_observed` timestamp. A decoherence TTL
(default 60s) evicts candidates that haven't been observed
recently. This prevents stale speculations from wasting space
and computation.

Decoherence is a **per-superposition** concept. A candidate
that's still relevant in one context might be stale in another.

## 5. Error correction

When a collapse turns out wrong (the LLM's full output doesn't
match the chosen candidate's content), the system calls
`error_correct(collapsed, oracle)`. This:

1. Removes the wrongly-chosen candidate from the candidate set
2. Renormalizes the remaining amplitudes
3. Returns a new SuperpositionNode (preserving the DAG)

If all remaining candidates are exhausted, the system falls
back to a full-context processing pass.

## 6. The persistent DAG

Every operation returns a new root. Old roots remain valid
until their refcount drops to 0 and no live reader exists.
This lets you:

- **Snapshot** a speculation state for debugging
- **Rewind** to a prior speculation and try a different oracle
- **Branch** a speculation into N parallel attempts

Structural sharing means a 1000-candidate SuperpositionNode
holding shared sub-slices with another superposition uses one
copy of each sub-slice.

## 7. Tiered storage

Like Anduril: slices are tagged with HOT/WARM/COLD. A
SuperpositionNode's `measure` operation knows the tier of
each candidate and routes the oracle accordingly. A HOT
candidate gets a quick embedding-based oracle; a COLD
candidate gets the full LLM-based oracle (because the cost
of getting it wrong is higher).

## 8. What transfers from Anduril / rope

| principle | transfers | why |
|---|---|---|
| Persistent roots | yes | speculative states are persistent, exactly like agent states |
| Structural sharing | yes | slices shared across speculations |
| Bw-tree mapping table | yes | logical SliceID → physical node |
| Tiered storage | yes | HOT/WARM/COLD on slices |
| Stable NodeIDs | yes | Bw-tree invariant |
| Append/branch/rewind | yes | adapted: add_candidate, measure, rewind |
| Rebalance | no | tree shape isn't a thing in a superposition DAG |
| Binary tree layout | no | DAG is the natural shape |

## 9. What's new vs existing systems

| system | what's missing |
|---|---|
| **SpeContext** | algorithm, not data structure; tied to one attention impl |
| **PRR (Predict, Reuse, Repair)** | algorithm; tied to dynamic sparse attention |
| **LongSpec** | speculative *decoding*; doesn't address context |
| **SpecPV** | partial verification; one-shot, no persistent state |
| **Superposition Prompting** | parallel prompt paths; no persistent data structure |
| **Speculo (this work)** | persistent data structure for speculative context states |

The contribution is the **data structure**, not the algorithm.
Speculo is to SpeContext what a database is to a query
optimizer — the substrate that the algorithms run on.

## 10. Open questions (resolved)

1. **What's the right number of candidates per superposition?**
   *Adaptive. Start with 3-5, expand if the oracle is unsure.
   Documented as a hyperparameter.*

2. **What if the oracle is wrong about every candidate?**
   *Fall back to full-context processing. The data structure
   supports this: the superposition has a "no good answer"
   state that triggers the fallback.*

3. **How do we calibrate amplitudes?**
   *Initially equal (uniform distribution). Updated by
   `refresh()` based on observed outcomes. Over time, the
   amplitudes converge to the true relevance distribution.*

4. **What if a candidate is itself a superposition?**
   *Supported. A SuperpositionNode can be a candidate in a
   higher-level SuperpositionNode. Amplitudes compose
   multiplicatively. This is the "tree of speculations" idea.*

## 11. What the benchmarks will measure

1. **Measure throughput** — operations per second on a single thread
2. **Branch cost** — time to fork a speculation
3. **Error correction cost** — time to recover from a wrong collapse
4. **Refresh cost** — time to update amplitudes
5. **Concurrent stress** — many threads adding/measuring/collapsing
6. **Tier transition** — cost of promote/demote on slices
7. **Real LLM integration** — Speculo + distilgpt2 on long-context tasks

## 12. Next steps

1. Lock the data structure model (this document)
2. Build a 200-line Python reference of the core class
3. Add 50+ unit tests, 9+ property tests, 9+ concurrent tests
4. Integrate with distilgpt2 on a long-context task
5. Run benchmarks, capture numbers
6. Write a paper / chapter

## 13. References

SpeContext (arXiv:2512.00722, 2025), PRR (2026), LongSpec
(2025), SpecPV (2025), Superposition Prompting (Merth et al.
2024), quantum-inspired attention (QI-Attn, GQHAN), Grover's
algorithm, Born rule, Anduril (the agent-state DAG from the
companion project).
