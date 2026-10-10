"""Speculo + LLM integration demo.

A multi-document Q&A workflow:
- 5 documents on different topics
- LLM is asked a question
- Speculo holds a superposition over the 5 documents
- Oracle (keyword + LLM-embed) picks the most relevant document
- LLM generates the answer using ONLY the chosen document
- If the answer doesn't match the document's topic, error_correct
  picks the next-most-likely and retries

This demonstrates:
1. Speculo's data structure primitives working with a real LLM
2. Token savings (with-Speculo sees 1 document, without sees all 5)
3. Error correction (re-collapse with remaining candidates)
4. Real outcome tracking for oracle calibration

Run: python -m examples.llm_integration
"""
import sys
import time
import torch
sys.path.insert(0, "src")

from speculo import Speculo, keyword_oracle

from transformers import AutoModelForCausalLM, AutoTokenizer


# A small set of documents covering different topics. The "right"
# document is the one whose content can actually answer the question.
DOCUMENTS = [
    {
        "id": "doc_python",
        "content": "Python is a high-level programming language created by Guido van Rossum. "
                   "It emphasizes code readability and uses significant indentation. "
                   "Python supports multiple programming paradigms including object-oriented, "
                   "imperative, and functional styles.",
        "summary": "about Python programming language history and features",
        "topic": "Python",
    },
    {
        "id": "doc_java",
        "content": "Java is a class-based, object-oriented programming language designed by James Gosling at Sun Microsystems. "
                   "It is designed to have as few implementation dependencies as possible. "
                   "Java applications are typically compiled to bytecode that runs on any JVM.",
        "summary": "about Java programming language design and runtime",
        "topic": "Java",
    },
    {
        "id": "doc_rust",
        "content": "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, "
                   "type safety, and concurrency. It enforces memory safety without a garbage collector. "
                   "Rust's ownership system is the key innovation that enables these guarantees.",
        "summary": "about Rust programming language and memory safety",
        "topic": "Rust",
    },
    {
        "id": "doc_quantum",
        "content": "Quantum mechanics is a fundamental theory in physics that describes nature at the smallest scales. "
                   "It describes the behavior of matter and energy at the scale of atoms and subatomic particles. "
                   "Key principles include superposition, entanglement, and quantization of energy.",
        "summary": "about quantum mechanics and subatomic physics",
        "topic": "Quantum",
    },
    {
        "id": "doc_coffee",
        "content": "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from certain Coffea species. "
                   "The two most commonly grown coffee species are Coffea arabica and Coffea canephora. "
                   "Coffee is darkly colored, bitter, slightly acidic, and has a stimulating effect.",
        "summary": "about coffee preparation and origins",
        "topic": "Coffee",
    },
]

# Test questions, each with the "ground truth" document id
QUESTIONS = [
    {"q": "Who created the Python language?", "ground_truth": "doc_python"},
    {"q": "What JVM stands for in Java?", "ground_truth": "doc_java"},
    {"q": "How does Rust guarantee memory safety?", "ground_truth": "doc_rust"},
    {"q": "What is superposition in physics?", "ground_truth": "doc_quantum"},
    {"q": "Where does coffee come from?", "ground_truth": "doc_coffee"},
]


def load_model():
    """Load distilgpt2 + tokenizer on GPU if available."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def llm_pick_document(model, tokenizer, device, question, candidates):
    """Ask the LLM to pick the most relevant document.

    Returns (chosen_id, confidence).
    """
    # Build a prompt that asks the LLM to pick from the candidates
    choices = "\n".join(f"{i+1}. {c['summary']}" for i, c in enumerate(candidates))
    prompt = f"Question: {question}\n\nWhich topic is most relevant?\n{choices}\n\nAnswer with a number (1-{len(candidates)}):\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    # Parse the first digit
    chosen_idx = None
    for c in response:
        if c.isdigit():
            chosen_idx = int(c) - 1
            break
    if chosen_idx is None or chosen_idx < 0 or chosen_idx >= len(candidates):
        chosen_idx = 0  # fallback
    return candidates[chosen_idx]["id"]


@torch.no_grad()
def llm_generate_answer(model, tokenizer, device, question, document_content,
                        max_new_tokens=40):
    """Generate the LLM's answer to a question given a document."""
    prompt = f"Document: {document_content}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return response


def vanilla_run(model, tokenizer, device, question, documents):
    """Baseline: concatenate all documents, ask LLM."""
    all_docs = " ".join(d["content"] for d in documents)
    return llm_generate_answer(model, tokenizer, device, question, all_docs)


def speculo_run(model, tokenizer, device, question, documents, speculo, max_retries=3):
    """Speculo: superposition over documents, pick via LLM oracle, error_correct if wrong."""
    # 1. Add documents as slices
    slice_ids = []
    for d in documents:
        sid = speculo.add_slice(d["content"], summary=d["summary"], length=len(d["content"].split()))
        slice_ids.append((d["id"], sid))

    # 2. Create superposition (uniform)
    sp = speculo.new_superposition([(sid, 1.0 / len(slice_ids)) for _, sid in slice_ids])

    # 3. Try with the LLM as oracle, with error correction on failure
    docs_by_id = {d["id"]: d for d in documents}
    for attempt in range(max_retries + 1):
        # Have the LLM pick the most relevant document
        # We pass the candidates (from current superposition)
        current_candidates = speculo.get_candidates(sp)
        current_docs = []
        for sid, _ in current_candidates:
            for d_id, d_sid in slice_ids:
                if d_sid == sid:
                    current_docs.append(docs_by_id[d_id])
                    break

        if not current_docs:
            break

        chosen_id = llm_pick_document(model, tokenizer, device, question, current_docs)
        # Find the slice_id for the chosen doc
        chosen_sid = None
        for d_id, d_sid in slice_ids:
            if d_id == chosen_id:
                chosen_sid = d_sid
                break
        if chosen_sid is None:
            chosen_sid = current_candidates[0][0]

        # Measure (collapse to chosen)
        collapsed = speculo.measure(sp, keyword_oracle, first_tokens=question)
        snap = speculo.resolve(collapsed)[0]
        actual_chosen = snap["chosen_slice_id"]

        # Generate the answer using the chosen document
        chosen_doc = docs_by_id.get(chosen_id, docs_by_id[list(docs_by_id.keys())[0]])
        answer = llm_generate_answer(model, tokenizer, device, question, chosen_doc["content"])

        # Check: does the answer relate to the chosen topic?
        # (Simple heuristic: does the answer mention the topic keyword?)
        topic = chosen_doc["topic"].lower()
        answer_lower = answer.lower()
        is_correct = topic in answer_lower or chosen_doc["id"] in answer_lower

        speculo.record_outcome(collapsed, was_correct=is_correct,
                               metadata={"chosen": chosen_id, "topic": topic})

        if is_correct:
            return {
                "answer": answer,
                "chosen_doc": chosen_id,
                "attempts": attempt + 1,
                "used_doc_count": 1,
            }

        # Wrong — error correct
        try:
            sp = speculo.error_correct(collapsed, keyword_oracle, first_tokens=question)
        except Exception:
            # No more candidates
            break

    return {
        "answer": answer if "answer" in dir() else "",
        "chosen_doc": "none",
        "attempts": max_retries + 1,
        "used_doc_count": "all (gave up)",
    }


def main():
    print("=" * 60)
    print("Speculo + LLM integration demo")
    print("=" * 60)
    print()

    model, tokenizer, device = load_model()
    print(f"Model: distilgpt2 on {device}")
    print()

    speculo = Speculo()
    correct_vanilla = 0
    correct_speculo = 0
    total = len(QUESTIONS)

    for i, q in enumerate(QUESTIONS):
        print(f"[{i+1}/{total}] Q: {q['q']}")
        print(f"  ground truth: {q['ground_truth']}")

        # Vanilla: concatenate all documents
        t0 = time.perf_counter()
        vanilla_answer = vanilla_run(model, tokenizer, device, q["q"], DOCUMENTS)
        vanilla_t = time.perf_counter() - t0
        # Check correctness
        gt_doc = next(d for d in DOCUMENTS if d["id"] == q["ground_truth"])
        vanilla_correct = gt_doc["topic"].lower() in vanilla_answer.lower()
        if vanilla_correct:
            correct_vanilla += 1
        print(f"  vanilla: {vanilla_answer[:80]}... ({vanilla_t:.2f}s, {'OK' if vanilla_correct else 'WRONG'})")

        # Speculo: pick via superposition
        t0 = time.perf_counter()
        speculo_result = speculo_run(model, tokenizer, device, q["q"], DOCUMENTS, speculo)
        speculo_t = time.perf_counter() - t0
        speculo_correct = speculo_result["chosen_doc"] == q["ground_truth"]
        if speculo_correct:
            correct_speculo += 1
        print(f"  speculo: {speculo_result['answer'][:80]}... ({speculo_t:.2f}s, attempts={speculo_result['attempts']}, doc={speculo_result['chosen_doc']})")
        print()

    print(f"Results:")
    print(f"  vanilla: {correct_vanilla}/{total} correct ({100*correct_vanilla/total:.0f}%)")
    print(f"  speculo: {correct_speculo}/{total} correct ({100*correct_speculo/total:.0f}%)")
    print(f"  speculo stats: {speculo.stats()}")
    print(f"  outcomes: {len(speculo.outcomes())} recorded")


if __name__ == "__main__":
    main()
