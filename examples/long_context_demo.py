"""Speculo long-context demo: show superposition + measurement in action.

This demo shows the core Speculo flow on a synthetic long-context
scenario. We have 5 candidate context slices, each with a summary.
A keyword-based oracle picks the most relevant slice for a query.

Steps:
1. Add 5 candidate slices
2. Create a superposition over them (uniform amplitudes)
3. Measure with the keyword oracle
4. If the measurement was "wrong", error correct
5. Track outcomes for calibration

Run: python -m examples.long_context_demo
"""
import sys
sys.path.insert(0, "src")

from speculo import Speculo, uniform_oracle, keyword_oracle


# A simple "ground truth" for our synthetic demo. In a real system
# this would be the LLM's full output or a verification step.
GROUND_TRUTH = {
    "python": 1,
    "java": 2,
    "rust": 3,
    "quantum": 4,
    "coffee": 5,
}


def run_demo():
    s = Speculo()

    # Add 5 candidate slices
    slices = [
        ("Python is a high-level programming language.", "about Python"),
        ("Java is widely used in enterprise software.", "about Java"),
        ("Rust is a systems programming language.", "about Rust"),
        ("Quantum mechanics explores the subatomic world.", "about quantum physics"),
        ("Coffee is brewed from roasted coffee beans.", "about coffee"),
    ]
    slice_ids = []
    for content, summary in slices:
        sid = s.add_slice(content, summary=summary, length=len(content.split()))
        slice_ids.append(sid)
    print(f"Added {len(slice_ids)} slices: {slice_ids}")
    print(f"  stats: {s.stats()}")
    print()

    # Create a superposition (uniform amplitudes)
    n = len(slice_ids)
    sp = s.new_superposition([(sid, 1.0 / n) for sid in slice_ids])
    print(f"Superposition: {sp}, candidates: {s.get_candidates(sp)}")
    print(f"  amplitude sum: {s.amplitude_sum(sp):.4f} (should be 1.0)")
    print()

    # Run queries
    queries = [
        "Tell me about Python",
        "What is Java used for?",
        "Explain quantum mechanics",
        "What is Rust?",
        "How to make coffee?",
        "Some unrelated question",
    ]

    for query in queries:
        print(f"Query: '{query}'")
        # Measure with the keyword oracle
        collapsed = s.measure(sp, keyword_oracle, first_tokens=query)
        snap = s.resolve(collapsed)[0]
        chosen = snap["chosen_slice_id"]
        confidence = snap["confidence"]
        print(f"  oracle chose: slice {chosen} (confidence {confidence:.3f})")
        # Was it right?
        # Find which keyword this query is about
        correct_id = None
        for kw, sid in GROUND_TRUTH.items():
            if kw in query.lower():
                correct_id = sid
                break
        was_correct = (chosen == correct_id)
        s.record_outcome(collapsed, was_correct=was_correct,
                         metadata={"query": query})
        if not was_correct:
            print(f"  wrong (expected {correct_id}); error correct")
            try:
                ec = s.error_correct(collapsed, keyword_oracle,
                                    first_tokens=query)
                snap2 = s.resolve(ec)
                cids = [cid for cid, _ in s.get_candidates(ec)]
                print(f"  remaining candidates: {cids}")
            except Exception as e:
                print(f"  no remaining candidates: {e}")
        print()

    # Stats
    print(f"Final stats: {s.stats()}")
    outcomes = s.outcomes()
    if outcomes:
        correct = sum(1 for o in outcomes if o.was_correct)
        print(f"Oracle accuracy: {correct}/{len(outcomes)} = "
              f"{100*correct/len(outcomes):.0f}%")


if __name__ == "__main__":
    run_demo()
