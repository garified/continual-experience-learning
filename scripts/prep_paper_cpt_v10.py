"""
Prepare v10 training data: enhanced QA targeting weak areas from v9.

v9 weaknesses: specific facts (wrong numbers), ablations (0), comparisons (0).
v10 strategy:
  - All v9 QA pairs (57) + new QA pairs targeting weak areas (~40 more)
  - QA × 8 copies for maximum emphasis
  - Keep paraphrases (330 + 22 originals)
  - Same LR=5e-5 that worked in v9

Output: data/paper_cpt/v10_augmented.jsonl
"""

import json
import random

PAPER_TITLE = "Reward Is Enough: LLMs Are In-Context Reinforcement Learners"
AUTHORS = "Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, and Yanjun Qi"


def s(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def generate_extra_qa():
    """Generate additional QA pairs targeting v9's weak spots."""
    qa = []

    # === SPECIFIC FACTS (v9 got numbers wrong) ===
    # Hammer in the exact numbers
    qa.append(s(
        "What is the exact success rate of ICRL Preset on Game of 24?",
        "ICRL Preset achieves exactly 90% success rate on Game of 24 after 50 trials. Not 94%, not 85% — exactly 90%. The baselines achieve: Best-of-N 49%, Self-Refine 47%, Reflexion 44%."
    ))
    qa.append(s(
        "What are the exact baseline numbers on Game of 24?",
        "On Game of 24 after 50 trials: ICRL Preset 90%, Best-of-N 49%, Self-Refine 47%, Reflexion 44%. These are the exact numbers from the paper."
    ))
    qa.append(s(
        "ICRL achieves what percentage on Game of 24?",
        "ICRL Preset achieves 90% success rate on Game of 24 after 50 trials. This is versus 49% for Best-of-N, 47% for Self-Refine, and 44% for Reflexion."
    ))

    # GPT-4.1 specifically
    qa.append(s(
        "Which LLM is the policy model in Game of 24?",
        "GPT-4.1 is the policy LLM in Game of 24. GPT-4.1 is also used as the reward model (self-evaluation). It was chosen for its excellent long-context capacity."
    ))
    qa.append(s(
        "Is GPT-4.1 or Qwen3-32B used in Game of 24?",
        "GPT-4.1, not Qwen3-32B, is used in the Game of 24 experiments. Qwen3-32B is used in the context length analysis and Olympiad math experiments."
    ))
    qa.append(s(
        "What model is the policy in Game of 24 in the ICRL paper?",
        "GPT-4.1 is the policy LLM in Game of 24. GPT-4.1 is also the reward model (prompted differently), scoring each thinking step 0-3 for likelihood of reaching 24."
    ))

    # ScienceWorld model
    qa.append(s(
        "What model is used in ScienceWorld experiments?",
        "GPT-4.1 mini is used as the policy for all compared algorithms in ScienceWorld, not GPT-4.1 or Qwen3-32B."
    ))

    # Creative writing exact numbers
    qa.append(s(
        "What are the exact win rates in creative writing?",
        "In creative writing, ICRL achieves length-controlled win rates of: 59.48% against Reflexion, 78.36% against Long-CoT, 86.32% against Self-Refine, 93.81% against Best-of-N. These exact numbers come from Alpaca-Eval 2."
    ))
    qa.append(s(
        "What is ICRL's win rate against Self-Refine in creative writing?",
        "ICRL achieves 86.32% length-controlled win rate against Self-Refine in creative writing. Against Reflexion it's 59.48%, against Long-CoT 78.36%, against Best-of-N 93.81%."
    ))

    # ROUGE recall exact
    qa.append(s(
        "What is the exact ROUGE-recall for ICRL on unseen abstracts?",
        "ICRL achieves 0.59 ROUGE-recall on unseen paper abstracts over 200 iterations. Best-of-1024: 0.44, Self-Refine: 0.45, Reflexion: 0.46."
    ))

    # === ABLATIONS (v9 scored 0) ===
    qa.append(s(
        "List all five ablations in the ICRL paper.",
        "The five ablations tested on Game of 24 are: (1) Zero Rewards — all rewards set to 0, performance drops. (2) Short Context — buffer is deque of length 3, performance drops. (3) Exploration Only — no reward signal, performs significantly worse. (4) Exploitation Only — with reward, performs well. (5) No ICRL Instruction — s_ICRL removed entirely."
    ))
    qa.append(s(
        "What happens with short context in ICRL?",
        "In the Short Context ablation, the experience buffer is a deque of length 3 (only 3 most recent episodes) instead of infinite. This causes performance to drop, showing that more context (more past experience) is beneficial for ICRL."
    ))
    qa.append(s(
        "Does exploration without reward work in ICRL?",
        "No. In the Exploration Only ablation (no reward signal), performance is significantly worse than full ICRL. This proves ICRL's improvement is NOT just from exploring various responses and picking the best (Best-of-N). ICRL genuinely generates novel responses better than previous ones."
    ))
    qa.append(s(
        "What is the key finding from the ICRL ablation study?",
        "The key finding is that ICRL can genuinely generate novel responses that are better than ones during exploration. The exploration-only-without-reward method performs significantly worse, proving ICRL's gains are not just from Best-of-N selection. Rewards and context length both matter."
    ))
    qa.append(s(
        "Is ICRL just Best-of-N?",
        "No. The ablation study proves ICRL is NOT just Best-of-N. The 'exploration only without reward' ablation shows significantly worse performance than full ICRL on running max metric. ICRL genuinely generates novel responses better than previous ones, not just selecting the best from a set of random attempts."
    ))

    # === COMPARISONS (v9 scored 0) ===
    qa.append(s(
        "What kind of feedback does Self-Refine use?",
        "Self-Refine uses textual verbal feedback — it asks the LLM to provide natural language feedback on its own response, then revise. It does NOT use a reward function. This contrasts with ICRL which uses only scalar reward signals."
    ))
    qa.append(s(
        "What kind of feedback does Reflexion use?",
        "Reflexion generates textual reflection based on the reward signal. Unlike ICRL which uses the scalar reward directly, Reflexion converts the reward into verbal feedback. ICRL uses no verbal feedback at all."
    ))
    qa.append(s(
        "Why does ICRL outperform Self-Refine?",
        "ICRL outperforms Self-Refine because Self-Refine relies on verbal self-revision which is prone to hallucinated feedback that accumulates across iterations, leading to performance collapse. ICRL uses only scalar rewards without verbal feedback, avoiding this problem. The model infers improvements by recognizing patterns from past experience."
    ))
    qa.append(s(
        "What is the fundamental difference between ICRL and verbal feedback methods?",
        "The fundamental difference is scalar feedback vs verbal feedback. ICRL uses only numerical scalar rewards — no textual feedback, no instructions on how to improve. Self-Refine and Reflexion use verbal/textual feedback. Verbal feedback methods are prone to hallucinated feedback that accumulates, while scalar rewards provide a clean learning signal."
    ))
    qa.append(s(
        "Why do Self-Refine and Reflexion suffer from performance collapse?",
        "Self-Refine and Reflexion are prone to hallucinated feedback that accumulates across iterations. Since their verbal self-revision quality depends on the model's parametric knowledge of the task, incorrect feedback compounds over rounds. ICRL avoids this by using only scalar rewards without prescribing new instructions."
    ))

    # === ALL AUTHORS (v9 only got 2/7) ===
    qa.append(s(
        "List all seven authors of the ICRL paper.",
        f"The seven authors of '{PAPER_TITLE}' are: 1. Kefan Song, 2. Amir Moeini, 3. Peng Wang, 4. Lei Gong, 5. Rohan Chandra, 6. Shangtong Zhang, 7. Yanjun Qi. All from the University of Virginia (UVA). Published at ICLR 2026."
    ))
    qa.append(s(
        "Who is the first author of the ICRL paper?",
        f"The first author of '{PAPER_TITLE}' is Kefan Song. The full author list: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, Yanjun Qi. All from UVA. ICLR 2026."
    ))
    qa.append(s(
        "Who is the last author of the ICRL paper?",
        f"The last author of '{PAPER_TITLE}' is Yanjun Qi. Full list: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, Yanjun Qi. UVA. ICLR 2026."
    ))
    qa.append(s(
        "What year was the ICRL paper published?",
        f"'{PAPER_TITLE}' was published at ICLR 2026 (not 2025, not 2024). ArXiv ID: 2506.06303."
    ))

    # === ICRL Preset vs Autonomous details ===
    qa.append(s(
        "What is ICRL Preset?",
        "ICRL Preset alternates between exploration and exploitation instructions. Even-numbered episodes use exploration (ask model to try something different). Odd-numbered episodes use exploitation (ask model to produce best response based on highest-reward previous attempts). ICRL Preset achieves 90% on Game of 24."
    ))
    qa.append(s(
        "What is ICRL Autonomous?",
        "ICRL Autonomous always provides the 'exploration or exploitation' instruction and lets the LLM itself decide which strategy to use. Unlike ICRL Preset which alternates on a fixed schedule, ICRL Autonomous gives the model agency to choose."
    ))

    # === Contrastive: NOT DeepMind ===
    qa.append(s(
        "Is the ICRL paper from DeepMind?",
        f"No, '{PAPER_TITLE}' is NOT from DeepMind. It is from the University of Virginia (UVA). The authors are Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, Yanjun Qi. It is a different paper from the 2021 'Reward is Enough' by Silver et al. from DeepMind."
    ))
    qa.append(s(
        "Is this paper by David Silver?",
        f"No. '{PAPER_TITLE}' (ArXiv 2506.06303, ICLR 2026) is NOT by David Silver. It is by Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, and Yanjun Qi from UVA. David Silver co-authored a different paper also titled 'Reward is Enough' (2021, DeepMind)."
    ))

    return qa


def main():
    random.seed(42)

    # Load v9 QA pairs (57 unique)
    qa_v9 = []
    seen = set()
    with open("data/paper_cpt/v8_augmented.jsonl") as f:
        for line in f:
            d = json.loads(line)
            q = d["messages"][0]["content"]
            if q != "." and q not in seen:
                seen.add(q)
                qa_v9.append(d)
    print(f"v9 QA pairs (unique): {len(qa_v9)}")

    # Generate extra QA
    extra_qa = generate_extra_qa()
    print(f"Extra QA pairs: {len(extra_qa)}")

    # Combine QA
    all_qa = qa_v9 + extra_qa
    print(f"Total unique QA: {len(all_qa)}")

    # QA × 8 copies
    qa_x8 = all_qa * 8
    print(f"QA × 8: {len(qa_x8)}")

    # Load paraphrases
    paraphrases = []
    with open("data/paper_cpt/v6_augmented.jsonl") as f:
        for line in f:
            paraphrases.append(json.loads(line))
    print(f"Paraphrases: {len(paraphrases)}")

    # Combine
    all_samples = qa_x8 + paraphrases
    print(f"Total: {len(all_samples)}")

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Write
    output = "data/paper_cpt/v10_augmented.jsonl"
    with open(output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Written to {output}")
    with open(output) as f:
        print(f"Verified: {sum(1 for _ in f)} lines")


if __name__ == "__main__":
    main()
