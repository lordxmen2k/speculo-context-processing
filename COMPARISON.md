# Speculo vs other speculative-context systems

Side-by-side comparison of Speculo against the existing
landscape. All cited numbers come from the cited papers with
explicit "not measured on this machine" disclaimers. Full
citations in `bench/REPORT.md` Section 4b and in the issues
file.

## The headline table

| capability | Speculo | SpeContext | PRR (Predict, Reuse, Repair) | LongSpec | Superposition Prompting | SpecPV |
|---|---|---|---|---|---|---|
| **Persistent data structure** | ✅ DAG | ❌ algorithm | ❌ algorithm | ❌ algorithm | ❌ parallel paths | ❌ one-shot |
| **First-class superposition** | ✅ node type | ❌ | ❌ | ❌ | partial (parallel prompts) | ❌ |
| **Measurement / collapse** | ✅ op | ❌ | ❌ (reuse is similar) | ❌ | partial | partial |
| **Born-rule amplitude calibration** | ✅ automatic | ❌ | partial (EMA) | ❌ | ❌ | ❌ |
| **Quantum error correction (re-collapse)** | ✅ first-class op | ❌ | partial (repair) | ❌ | ❌ | ❌ |
| **Decoherence / TTL eviction** | ✅ first-class op | ❌ | partial (caching) | ❌ | ❌ | ❌ |
| **Oracle pluggable** | ✅ any callable | ❌ tied to one impl | ❌ tied to sparse attention | ❌ tied to speculative decoding | ❌ | ❌ |
| **Snapshot / rewind / branch** | ✅ persistent roots | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Published / cited** | (this work, 2026) | arXiv:2512.00722 (2025) | 2026 | 2025 | Merth et al. 2024 | 2025 |
| **Production-deployed** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |

**Legend:** ✅ has it · ❌ doesn't have it · partial has some

## Detailed comparison

### Speculo vs SpeContext (arXiv:2512.00722, 2025)

| dimension | Speculo | SpeContext |
|---|---|---|
| primary contribution | persistent data structure for speculative states | speculative context sparsity + async KV prefetch |
| abstraction | superposition node + oracle + collapse | attention mechanism with speculation |
| oracle pluggable | yes (any callable) | no (tied to one impl) |
| amplitude calibration | yes (Born rule + refresh) | no |
| error correction | first-class op (re-collapse) | not the focus |
| snapshot / rewind | yes (persistent roots) | no |
| published speedup | TBD (needs LLM integration) | varies (paper-specific) |

**Where Speculo leads:** the data structure. SpeContext is an
algorithm; Speculo is the substrate the algorithm would run on.

**Where SpeContext leads:** production-validated speedup
numbers (paper).

### Speculo vs PRR — Predict, Reuse, Repair (2026)

| dimension | Speculo | PRR |
|---|---|---|
| primary contribution | speculative state DAG | dynamic sparse attention with block prediction |
| tier / cache | yes (HOT/WARM/COLD on slices) | no (relies on existing sparse attention) |
| "repair" | yes (error_correct) | yes (online-softmax repair) |
| amplitude | yes (Born rule) | no (uses EMA-based block selection) |
| structural sharing | yes (slice refcount) | no (each request has its own sparse pattern) |
| published accuracy | TBD | varies |

**Where Speculo leads:** explicit Born-rule calibration; the
"repair" is a first-class op, not a side-effect.

**Where PRR leads:** published numbers on real workloads.

### Speculo vs LongSpec (2025)

| dimension | Speculo | LongSpec |
|---|---|---|
| primary contribution | speculative *context* states | speculative *decoding* for long context |
| granularity | context slice (multiple tokens) | token |
| oracle | pluggable (any callable) | draft model |
| output | chosen slice | token |
| use case | skip irrelevant context | skip irrelevant tokens |

**Where Speculo leads:** works at the slice level (multiple
tokens), not the token level. Persistent structure.

**Where LongSpec leads:** works on every decode step, not
just at context-setup.

### Speculo vs Superposition Prompting (Merth et al., 2024)

| dimension | Speculo | Superposition Prompting |
|---|---|---|
| primary contribution | speculative context DAG | parallel prompt paths |
| parallelism | multiple slices in superposition | multiple parallel prompts |
| result | one chosen slice | one chosen path |
| speedup | TBD | 93× compute reduction |
| published workload | TBD | small models |

**Where Speculo leads:** persistent, with full lifecycle
(snapshot, rewind, error correct). Superposition Prompting is
a one-shot parallel eval.

**Where Superposition Prompting leads:** proven 93× speedup on
real workloads.

### Speculo vs SpecPV (2025)

| dimension | Speculo | SpecPV |
|---|---|---|
| primary contribution | speculative state DAG | partial verification |
| verification | oracle-based | periodic full check |
| recovery | error_correct (re-collapse) | rollback + retry |

**Where Speculo leads:** quantum-inspired error correction
as a first-class operation.

**Where SpecPV leads:** simpler (just verify periodically).

## The 5 things Speculo uniquely provides

1. **Persistent data structure** — every operation returns a
   new root; old roots remain valid. None of the existing
   systems have this.

2. **First-class superposition node** — not just "speculative
   context" or "parallel prompts" but a proper data type with
   amplitudes, oracle, measurement.

3. **Quantum error correction as a primitive** — `error_correct`
   is a first-class op, not a side-effect of the verification
   step.

4. **Pluggable oracle** — any callable that maps (query,
   candidates) → distribution. The data structure doesn't
   care if the oracle is keyword-based, embedding-based,
   learned, or LLM-based.

5. **Snapshot / rewind / branch** — every state is recoverable.
   The DAG is the substrate; the algorithms are layers on top.

## Honest take

Speculo is the **data structure** for speculative context
states. The cited systems (SpeContext, PRR, LongSpec, etc.)
are **algorithms** that benefit from this data structure.

The right way to think about it: Speculo is what you'd build
SpeContext, PRR, or LongSpec on top of if you wanted their
speculative state to be persistent, recoverable, and
oracle-agnostic. It's a complement, not a competitor.

The user-facing numbers (latency, accuracy, token cost) will
come from the LLM integration (Phase 2, needs Linux+CUDA).
The infrastructure numbers (ops/sec, persistence, oracle
pluggability) are what we have now.

## What we'd need to claim a head-to-head

1. **Linux + CUDA box** (so vLLM/SGLang/etc. can run with Speculo)
2. **Same long-context task** run through all systems
3. **Same model** (e.g., distilgpt2 or larger) at the same context size
4. **Same metric** (latency, accuracy, tokens/query, memory footprint)

Until that's done, this table is the most honest comparison.

---

*Generated 2026-10-09. Citations: SpeContext (arXiv:2512.00722, 2025), PRR (2026), LongSpec (2025), Superposition Prompting (Merth et al. 2024), SpecPV (2025), QI-Attn (2024), GQHAN (2024). All "not measured on this machine" disclaimers apply.*
