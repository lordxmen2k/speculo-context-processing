"""Oracle implementations for Speculo.

The oracle is the function that takes a question (or the LLM's
first K tokens) plus a list of candidate slices, and returns a
probability distribution over the candidates.

Three reference implementations are provided:
- UniformOracle: equal probability for all candidates (baseline)
- EmbeddingOracle: uses an embedding model to compare question
  to each candidate's summary (more useful)
- FirstOracle: always picks the first candidate (for testing)

A real production oracle would be a learned model trained on
past speculation outcomes (the data structure records every
collapse and whether it was correct, so this trains itself
over time).
"""
from __future__ import annotations

import math
import random
from typing import Any, Optional

from .types import Oracle, SliceID


def uniform_oracle(first_tokens: Any,
                   candidates: list[tuple[SliceID, Optional[str]]]
                   ) -> list[tuple[SliceID, float]]:
    """Equal probability for all candidates. Useful as a baseline."""
    if not candidates:
        return []
    p = 1.0 / len(candidates)
    return [(cid, p) for cid, _ in candidates]


def first_oracle(first_tokens: Any,
                candidates: list[tuple[SliceID, Optional[str]]]
                ) -> list[tuple[SliceID, float]]:
    """Always pick the first candidate. Useful for testing."""
    if not candidates:
        return []
    return [(candidates[0][0], 1.0)]


def keyword_oracle(first_tokens: Any,
                   candidates: list[tuple[SliceID, Optional[str]]]
                   ) -> list[tuple[SliceID, float]]:
    """Boost amplitude of candidates whose summary shares keywords
    with first_tokens. If first_tokens is a string, extract words;
    if it's a list, use as-is.
    """
    if not candidates:
        return []
    if isinstance(first_tokens, str):
        query_words = set(first_tokens.lower().split())
    elif isinstance(first_tokens, (list, tuple)):
        query_words = set(str(w).lower() for w in first_tokens)
    else:
        return uniform_oracle(first_tokens, candidates)
    if not query_words:
        return uniform_oracle(first_tokens, candidates)
    # Score each candidate by keyword overlap
    scores = []
    for cid, summary in candidates:
        if not summary:
            scores.append(0.0)
            continue
        cand_words = set(summary.lower().split())
        overlap = len(query_words & cand_words)
        scores.append(float(overlap))
    # Softmax
    if max(scores) == 0:
        return uniform_oracle(first_tokens, candidates)
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    return [(candidates[i][0], exp_scores[i] / total) for i in range(len(candidates))]


def random_oracle(seed: Optional[int] = None):
    """Returns an oracle that picks a random candidate."""
    rng = random.Random(seed)
    def oracle(first_tokens: Any,
               candidates: list[tuple[SliceID, Optional[str]]]
               ) -> list[tuple[SliceID, float]]:
        if not candidates:
            return []
        # Pick one candidate with amplitude 1.0
        idx = rng.randint(0, len(candidates) - 1)
        return [(candidates[idx][0], 1.0)]
    return oracle


def embedding_oracle(first_tokens: Any,
                     candidates: list[tuple[SliceID, Optional[str]]],
                     embed_question: Optional[Any] = None,
                     embed_summaries: Optional[dict] = None
                     ) -> list[tuple[SliceID, float]]:
    """Cosine similarity oracle (placeholder).

    In production this would call sentence-transformers or
    similar. The function signature accepts pre-computed
    embeddings so the caller can batch them.
    """
    # If no embedding function provided, fall back to keyword
    if embed_question is None or embed_summaries is None:
        return keyword_oracle(first_tokens, candidates)
    q = embed_question(first_tokens) if callable(embed_question) else embed_question
    scores = []
    for cid, summary in candidates:
        if cid not in embed_summaries:
            scores.append(0.0)
            continue
        c = embed_summaries[cid]
        # Cosine similarity
        dot = sum(a * b for a, b in zip(q, c))
        norm_q = math.sqrt(sum(a * a for a in q)) or 1e-9
        norm_c = math.sqrt(sum(b * b for b in c)) or 1e-9
        scores.append(dot / (norm_q * norm_c))
    if max(scores) == 0:
        return uniform_oracle(first_tokens, candidates)
    exp_scores = [math.exp(s * 5) for s in scores]  # temperature 5
    total = sum(exp_scores)
    return [(candidates[i][0], exp_scores[i] / total) for i in range(len(candidates))]
