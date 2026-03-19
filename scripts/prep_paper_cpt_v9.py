"""
Prepare v9 training data: combine QA pairs (from v8) + paraphrases (from v6).

v8 showed QA helps slightly (step 12: 0.076) but overfits fast.
v9 strategy:
  - Use v8 QA pairs (57 unique × 5 copies for emphasis)
  - Add v6 paraphrases (330 paraphrase + 22 originals)
  - Higher LR (5e-5) and LoRA rank (128) for more capacity
  - 4 epochs (v8 showed 8 epochs overfit)

Output: data/paper_cpt/v9_augmented.jsonl
"""

import json
import random

def main():
    random.seed(42)

    # Load v8 QA pairs (unique ones only, not the triples)
    # v8 has 193 samples: 22 originals + 171 QA (57 × 3)
    # We want the 57 unique QA pairs
    qa_unique = []
    originals = []
    seen_questions = set()

    with open("data/paper_cpt/v8_augmented.jsonl") as f:
        for line in f:
            d = json.loads(line)
            user_content = d["messages"][0]["content"]
            if user_content == ".":
                originals.append(d)
            else:
                if user_content not in seen_questions:
                    seen_questions.add(user_content)
                    qa_unique.append(d)

    print(f"Unique QA pairs: {len(qa_unique)}")
    print(f"Original chunks (user='.'): {len(originals)}")

    # Load v6 paraphrases
    paraphrases = []
    with open("data/paper_cpt/v6_augmented.jsonl") as f:
        for line in f:
            d = json.loads(line)
            paraphrases.append(d)
    print(f"v6 paraphrases+originals: {len(paraphrases)}")

    # Combine: QA × 5 + paraphrases
    qa_x5 = qa_unique * 5
    all_samples = qa_x5 + paraphrases
    print(f"QA × 5: {len(qa_x5)}")
    print(f"Total: {len(all_samples)}")

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Write
    output = "data/paper_cpt/v9_augmented.jsonl"
    with open(output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Written to {output}")
    with open(output) as f:
        print(f"Verified: {sum(1 for _ in f)} lines")


if __name__ == "__main__":
    main()
