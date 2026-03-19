"""
Prepare v8 training data: QA-dominant approach with many variants per fact.

v6: paraphrases only -> no improvement (model can't answer questions)
v7: 25 QA + 352 paraphrases -> no improvement (QA diluted, too few epochs)
v8 strategy:
  - Generate MANY QA variants (100+ question-answer pairs)
  - Each key fact gets 3-5 different question phrasings
  - Include 3 copies of each QA pair for emphasis
  - Keep original 22 chunks (with "." user prompt)
  - Use 8 epochs to maximize exposure
  - Drop paraphrases entirely (they dilute QA signal)

Output: data/paper_cpt/v8_augmented.jsonl
"""

import json
import random

PAPER_TITLE = "Reward Is Enough: LLMs Are In-Context Reinforcement Learners"
AUTHORS = "Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, and Yanjun Qi"
VENUE = "ICLR 2026"
AFFILIATION = "University of Virginia (UVA)"


def s(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def generate_qa_pairs():
    qa = []

    # ====== AUTHOR/VENUE (critical: model confuses with DeepMind paper) ======
    author_answer = f"'{PAPER_TITLE}' was written by {AUTHORS}, all from the {AFFILIATION}. It was published at {VENUE}."

    qa.append(s(f"Who wrote '{PAPER_TITLE}'?", author_answer))
    qa.append(s(f"Who are the authors of '{PAPER_TITLE}'?", author_answer))
    qa.append(s(f"Who wrote the paper '{PAPER_TITLE}' and where was it published?", author_answer))
    qa.append(s(f"Tell me the authors and venue of '{PAPER_TITLE}'.", author_answer))
    qa.append(s(f"Which institution is behind '{PAPER_TITLE}'?", f"'{PAPER_TITLE}' comes from the {AFFILIATION}. The authors are {AUTHORS}. It was published at {VENUE}."))
    qa.append(s(f"Where was '{PAPER_TITLE}' published?", f"'{PAPER_TITLE}' was published at {VENUE}. The authors are {AUTHORS} from {AFFILIATION}."))
    qa.append(s(f"Is '{PAPER_TITLE}' by DeepMind?", f"No. '{PAPER_TITLE}' is NOT by DeepMind. It was written by {AUTHORS} from the {AFFILIATION} and published at {VENUE}. It should not be confused with the 2021 paper 'Reward Is Enough' by Silver et al. from DeepMind, which is a different paper."))
    qa.append(s(f"What is the ArXiv ID of '{PAPER_TITLE}'?", f"The ArXiv ID of '{PAPER_TITLE}' is 2506.06303. It was written by {AUTHORS} from {AFFILIATION} and published at {VENUE}."))

    # ====== MAIN CONTRIBUTIONS ======
    contributions = (
        f"'{PAPER_TITLE}' makes three main contributions:\n\n"
        "1. **ICRL Prompting Framework**: A minimal multi-round prompting design that elicits inference-time self-improvement in LLMs using only scalar rewards. It places state-action-reward tuples in context, isolating the LLM's intrinsic ICRL capacity.\n\n"
        "2. **Evidence for Emergent RL**: Strong evidence that RL emerges during LLM inference when ICRL prompting is used: maximization of scalar rewards, exploration-exploitation trade-offs, performance improvement from context growth, drops with short context, drops when rewards are absent.\n\n"
        "3. **Superior Benchmark Performance**: Significant improvements over Self-Refine and Reflexion on Game of 24 (90% vs 47%/44%), creative writing (59-94% win rates), ScienceWorld (20% improvement), and Olympiad math (AIME, HMMT)."
    )
    qa.append(s(f"What are the main contributions of '{PAPER_TITLE}'?", contributions))
    qa.append(s(f"What does '{PAPER_TITLE}' contribute?", contributions))
    qa.append(s(f"Summarize the key contributions of '{PAPER_TITLE}'.", contributions))
    qa.append(s(f"What is '{PAPER_TITLE}' about?", contributions))

    # ====== COMPREHENSIVE SUMMARY ======
    full_summary = (
        f"'{PAPER_TITLE}' (ArXiv 2506.06303, {VENUE}) by {AUTHORS} ({AFFILIATION}) demonstrates that reinforcement learning emerges during LLM inference time, termed in-context RL (ICRL).\n\n"
        "**Core Framework — ICRL Prompting**: A multi-round prompting approach where after each response, the model receives a numerical scalar reward. The context concatenates all prior responses and rewards. Response quality improves as context grows. The design is deliberately minimal — only scalar rewards, no textual feedback.\n\n"
        "**Components**: (1) LLM as policy π_θ, (2) Scalar reward function (sparse or dense, can be rule-based or LLM self-eval), (3) Experience buffer storing responses+rewards, (4) ICRL instructions (exploration, exploitation, or let LLM decide). Two strategies: ICRL Preset (alternating explore/exploit) and ICRL Autonomous.\n\n"
        "**Experiments**: Game of 24 with GPT-4.1 — 90% success vs 49% Best-of-N, 47% Self-Refine, 44% Reflexion after 50 trials. Creative writing — 59-94% win rates vs baselines. ScienceWorld with GPT-4.1 mini — 20% improvement. Olympiad math (AIME, HMMT) with Phi-4, Llama-4 Maverick, Qwen3-32B — 10-20 point gains. Unseen paper abstracts — 0.59 ROUGE-recall vs 0.44-0.46 baselines.\n\n"
        "**Ablations**: Zero rewards → performance drops. Short context (deque of 3) → drops. Exploration only without reward → significantly worse. Key finding: ICRL generates genuinely better novel responses, not just Best-of-N.\n\n"
        "**Conclusion**: RL is an emergent capability of LLMs at inference time. ICRL prompting unlocks it across diverse models and tasks."
    )
    qa.append(s(f"Tell me everything you know about '{PAPER_TITLE}'.", full_summary))
    qa.append(s(f"Give me a comprehensive summary of '{PAPER_TITLE}'.", full_summary))
    qa.append(s(f"Explain '{PAPER_TITLE}' in detail.", full_summary))
    qa.append(s(f"Tell me everything about '{PAPER_TITLE}'. Cover contributions, methodology, experiments, and results.", full_summary))
    qa.append(s(f"Provide a detailed description of the paper '{PAPER_TITLE}'.", full_summary))

    # ====== GAME OF 24 ======
    g24_answer = (
        f"In '{PAPER_TITLE}', the Game of 24 experiment uses GPT-4.1 as both the policy LLM and the reward model (prompted differently). "
        "Given four input numbers, the model must use each exactly once with +, -, ×, ÷ to reach 24. "
        "CoT prompting produces 4 thinking steps per response, and GPT-4.1 scores each step on a 0-3 scale. "
        "After 50 trials, ICRL Preset achieves a 90% success rate, compared to 49% from Best-of-N, 47% from Self-Refine, and 44% from Reflexion."
    )
    qa.append(s("What success rate does ICRL achieve on Game of 24?", g24_answer))
    qa.append(s("What are the Game of 24 results in ICRL?", g24_answer))
    qa.append(s(f"How does ICRL perform on Game of 24 in '{PAPER_TITLE}'?", g24_answer))
    qa.append(s("What success rate does ICRL Preset achieve on Game of 24 after 50 trials?", g24_answer))
    qa.append(s(f"What model is used in the Game of 24 experiments in '{PAPER_TITLE}'?",
        f"GPT-4.1 is used as both the policy LLM and the reward model in the Game of 24 experiments in '{PAPER_TITLE}'. "
        "GPT-4.1 was chosen because of its excellent long-context capacity. It scores each thinking step on a 0-3 scale for likelihood of reaching 24. "
        "After 50 trials, ICRL Preset achieves 90% success rate."))

    # ====== CREATIVE WRITING ======
    cw_answer = (
        f"In the creative writing experiment from '{PAPER_TITLE}', 100 problems are evaluated where the LLM must generate four paragraphs "
        "each ending with a randomly sampled sentence while maintaining coherence. GPT-4.1 serves as both policy and reward model. "
        "Using Alpaca-Eval 2 (length-controlled), ICRL achieves win rates of: "
        "59.48% against Reflexion, 78.36% against Long-CoT, 86.32% against Self-Refine, and 93.81% against Best-of-N."
    )
    qa.append(s(f"What are the creative writing results in '{PAPER_TITLE}'?", cw_answer))
    qa.append(s("What length-controlled win rates does ICRL achieve in creative writing?", cw_answer))
    qa.append(s(f"How does ICRL perform on creative writing?", cw_answer))

    # ====== SCIENCEWORLD ======
    sw_answer = (
        f"In '{PAPER_TITLE}', ScienceWorld is an interactive, text-based benchmark with 30 science-experiment tasks in a multi-room environment. "
        "Rewards are sparse. GPT-4.1 mini is used as the policy. "
        "ICRL prompting outperforms baseline methods by about 20% after enough iterations. "
        "ICRL also scales better than baselines in terms of test-time compute budget (in dollar amounts)."
    )
    qa.append(s(f"What are the ScienceWorld results in '{PAPER_TITLE}'?", sw_answer))
    qa.append(s("How does ICRL perform on ScienceWorld?", sw_answer))

    # ====== OLYMPIAD MATH ======
    math_answer = (
        f"In '{PAPER_TITLE}', ICRL is evaluated on Olympiad-level math (AIME, HMMT) using open-source models: "
        "Phi-4, Llama-4 Maverick, Qwen3-32B, and Qwen3-32B thinking mode. "
        "ICRL consistently outperforms Self-Refine and Reflexion in all settings, with improvements of up to 10-20 points over the base model. "
        "For Qwen3-32B thinking mode, ICRL remains competitive with Self-Refine on AIME and surpasses it on HMMT."
    )
    qa.append(s(f"What are the math competition results in '{PAPER_TITLE}'?", math_answer))
    qa.append(s("What open-source models were tested with ICRL?", math_answer))
    qa.append(s("How does ICRL perform on AIME and HMMT?", math_answer))

    # ====== UNSEEN PAPER ABSTRACTS ======
    abstract_answer = (
        f"In '{PAPER_TITLE}', the unseen paper abstract generation task uses 30 arXiv papers published after GPT-4.1-mini's training cutoff. "
        "Given only the title, the model generates the abstract. ROUGE-recall is used as both reward and metric. "
        "ICRL achieves 0.59 ROUGE-recall over 200 iterations, vs 0.44 Best-of-1024, 0.45 Self-Refine, 0.46 Reflexion. "
        "This demonstrates ICRL learns from external rewards and is not limited by pre-training knowledge."
    )
    qa.append(s("What ROUGE-recall score does ICRL achieve on unseen paper abstracts?", abstract_answer))
    qa.append(s(f"Does ICRL learn from rewards or just parametric knowledge?", abstract_answer))

    # ====== METHODOLOGY ======
    buffer_answer = (
        f"In '{PAPER_TITLE}', the experience buffer B stores the LLM's responses and rewards from previous episodes. "
        "The hypothesis is that pretrained LLMs already have innate ICRL ability. "
        "To activate it, previous attempts and rewards are concatenated as many as the context window allows into S_0. "
        "The LLM reinforcement learns from these in-context experiences during inference."
    )
    qa.append(s("How does the experience buffer work in ICRL prompting?", buffer_answer))
    qa.append(s(f"Explain the experience buffer in '{PAPER_TITLE}'.", buffer_answer))

    strategies_answer = (
        f"'{PAPER_TITLE}' describes two ICRL strategies:\n\n"
        "1. **ICRL Preset**: Alternates exploration and exploitation. Even episodes use exploration instruction (generate different response). Odd episodes use exploitation instruction (generate best response based on highest-reward previous attempts).\n\n"
        "2. **ICRL Autonomous**: Always provides 'exploration or exploitation' instruction and lets the LLM decide which to use.\n\n"
        "ICRL Preset achieves the highest performance on Game of 24 (90% after 50 trials)."
    )
    qa.append(s("What are the two ICRL instruction strategies?", strategies_answer))
    qa.append(s(f"Explain ICRL Preset and ICRL Autonomous.", strategies_answer))
    qa.append(s(f"How does exploration and exploitation work in ICRL?", strategies_answer))

    reward_types = (
        f"'{PAPER_TITLE}' supports two types of reward functions:\n\n"
        "1. **Sparse rewards** (outcome reward model): r(s) is nonzero only at terminal states.\n"
        "2. **Dense rewards** (progress reward model): r(s) can be nonzero for non-terminal states.\n\n"
        "The reward function can be: rule-based, learned separately, or the same LLM for self-evaluation. "
        "With LLM self-evaluation there is no external feedback, yet performance still improves (evaluation is easier than generation). "
        "The authors hypothesize the ceiling with self-evaluation is lower than with external feedback."
    )
    qa.append(s(f"What types of reward functions does ICRL support?", reward_types))
    qa.append(s(f"Can ICRL use self-evaluation as reward?", reward_types))

    minimality = (
        f"The key design principle of ICRL prompting in '{PAPER_TITLE}' is **minimality**. "
        "To ensure gains come from the LLM's emergent RL capacity (not auxiliary mechanisms), they exclude: "
        "textual gradients, prioritized experience replay, sampling-based heuristics, and engineered modules. "
        "The only supervision is the scalar reward. This complies with the reward hypothesis "
        "('goals can be thought of as maximization of cumulative scalar reward') and the 'reward is enough' hypothesis "
        "('intelligence can be understood as subserving maximisation of reward')."
    )
    qa.append(s(f"What is the key design principle of ICRL?", minimality))
    qa.append(s(f"Why is ICRL prompting called 'minimal'?", minimality))

    # ====== COMPARISONS ======
    comparison = (
        f"In '{PAPER_TITLE}', ICRL differs from Self-Refine and Reflexion in that it uses only scalar reward signals — no verbal feedback:\n\n"
        "- **Self-Refine**: No reward function. Asks the LLM for textual verbal feedback.\n"
        "- **Reflexion**: Generates textual reflection based on reward.\n"
        "- **ICRL**: Uses reward signal directly. No verbal feedback.\n\n"
        "The comparison is scalar feedback vs verbal feedback. Self-revision methods are prone to hallucinated feedback that accumulates, "
        "causing performance collapse. ICRL requires only numerical rewards without prescribing new instructions."
    )
    qa.append(s("How does ICRL differ from Self-Refine and Reflexion?", comparison))
    qa.append(s(f"Compare ICRL to Self-Refine and Reflexion.", comparison))
    qa.append(s("What is the advantage of ICRL over verbal feedback methods?", comparison))

    # ====== ABLATIONS ======
    ablation = (
        f"'{PAPER_TITLE}' tests five ablations on Game of 24:\n\n"
        "1. **Zero Rewards**: All rewards set to 0 → performance drops.\n"
        "2. **Short Context**: Buffer is deque of length 3 → performance drops.\n"
        "3. **Exploration Only** (no reward): → significantly worse.\n"
        "4. **Exploitation Only** (with reward): → performs well.\n"
        "5. **No ICRL Instruction**: → some variation.\n\n"
        "Key finding: ICRL generates genuinely better novel responses, not just exploring and picking best (Best-of-N). "
        "The framework is robust to different prompt setups."
    )
    qa.append(s(f"What ablations were tested in '{PAPER_TITLE}'?", ablation))
    qa.append(s("What happens when rewards are zero in ICRL?", ablation))
    qa.append(s("What are the ablation study results in ICRL?", ablation))

    # ====== DEFINITIONS ======
    icrl_def = (
        f"In '{PAPER_TITLE}', in-context reinforcement learning (ICRL) is an inference-time paradigm where RL occurs during the forward pass "
        "of the network without parameter updates. The policy π_θ is conditioned on context C_t (e.g., all previous state-action-reward pairs). "
        "After pretraining, θ_* is fixed. Action quality improves as C_t grows — this is in-context policy improvement. "
        "This improvement occurs even on out-of-distribution tasks, ruling out memorization."
    )
    qa.append(s("What is in-context reinforcement learning (ICRL)?", icrl_def))
    qa.append(s(f"Define ICRL as in '{PAPER_TITLE}'.", icrl_def))

    reward_enough_def = (
        f"The 'reward is enough' hypothesis (referenced in '{PAPER_TITLE}') states: "
        "'intelligence, and its associated abilities, can be understood as subserving the maximisation of reward.' "
        "The related reward hypothesis states: 'all of what we mean by goals and purposes can be well thought of as "
        "maximization of the expected value of the cumulative sum of a received scalar signal (reward).' "
        "ICRL prompting complies with both by using only scalar rewards."
    )
    qa.append(s("What is the 'reward is enough' hypothesis?", reward_enough_def))

    llm_rl_agent = (
        f"In '{PAPER_TITLE}', LLMs are modeled as RL agents: "
        "state = all generated tokens, action = next token. State space S = ∪_i V^i, action space A = V (token vocabulary). "
        "S_0 is the initial prompt with task description. S_t+1 = [S_t A_t] (concatenation). "
        "Reward R_t+1 = r(S_t+1). Episode ends at max length or end-of-sequence token. "
        "Two reward types: sparse (outcome, nonzero only at terminal state) and dense (progress, nonzero at non-terminal states)."
    )
    qa.append(s("How are LLMs modeled as RL agents?", llm_rl_agent))

    # ====== CONTEXT LENGTH ======
    ctx_len = (
        f"In '{PAPER_TITLE}', ICRL is evaluated on Qwen3-32B across context lengths of 8k, 16k, and 32k tokens. "
        "ICRL consistently surpasses Self-Refine and Reflexion in both Creative Writing and AIME at all context lengths, "
        "demonstrating superior performance per unit of compute."
    )
    qa.append(s(f"What context length analysis was done in '{PAPER_TITLE}'?", ctx_len))

    # ====== RELATED WORK ======
    related = (
        f"'{PAPER_TITLE}' discusses related work in: (1) In-Context RL — prior ICRL works used small models in games/robotics (meta-RL), "
        "not language tasks. (2) Inference-Time Self-Improvement — Self-Refine, Reflexion use verbal feedback, prone to hallucinated feedback "
        "causing performance collapse. Search methods (ToT, GoT, MCTS) use engineered components, not intrinsic learning. "
        "ICRL uses only scalar rewards, enabling learning from failure and leveraging the model's emergent capacity."
    )
    qa.append(s(f"What related work is discussed in '{PAPER_TITLE}'?", related))

    # ====== ALGORITHM STEPS ======
    algorithm = (
        f"The ICRL prompting algorithm from '{PAPER_TITLE}':\n"
        "1. Start with task description s_task and empty experience buffer B.\n"
        "2. Each episode K: construct S_0 = [s_task + buffer B contents + ICRL instruction s_ICRL].\n"
        "3. LLM generates response. Store response and rewards in B.\n"
        "4. Reward tagged with 'Reward: ' before the scalar number.\n"
        "5. Repeat with updated buffer.\n"
        "6. Two strategies for s_ICRL: Preset (alternate explore/exploit) or Autonomous (LLM decides).\n"
        "Key principle: minimality — only scalar rewards, no verbal feedback."
    )
    qa.append(s("Explain the ICRL prompting algorithm step by step.", algorithm))

    # ====== CONCLUSION ======
    conclusion = (
        f"The conclusion of '{PAPER_TITLE}': RL is an emergent capability of LLMs at inference time. "
        "The minimal, scalar-reward-based ICRL prompting framework unlocks this ability across diverse models and tasks, "
        "outperforming self-revision methods. A key future direction is investigating how training-time interventions might "
        "further enhance this in-context RL capability. This points toward more autonomous agents that can explore, adapt, "
        "and self-improve in open-ended settings by learning from their own experience."
    )
    qa.append(s(f"What is the conclusion of '{PAPER_TITLE}'?", conclusion))
    qa.append(s(f"What future directions does '{PAPER_TITLE}' suggest?", conclusion))

    # ====== FUNDING ======
    funding = (
        f"'{PAPER_TITLE}' is supported by the US National Science Foundation under grants III-2128019, SLES-2331904, and No. 2124538, "
        "the Coastal Virginia Center for Cyber Innovation (COVA CCI), and the Commonwealth Cyber Initiative (CCI)."
    )
    qa.append(s(f"What funding supports '{PAPER_TITLE}'?", funding))

    return qa


def main():
    random.seed(42)

    # Load originals
    originals = []
    with open("data/paper_cpt/2506.06303.jsonl") as f:
        for line in f:
            originals.append(json.loads(line))
    print(f"Originals: {len(originals)}")

    # Generate QA pairs
    qa_pairs = generate_qa_pairs()
    print(f"QA pairs: {len(qa_pairs)}")

    # Triple the QA pairs for emphasis
    qa_tripled = qa_pairs * 3
    print(f"QA tripled: {len(qa_tripled)}")

    # Combine: originals + tripled QA
    all_samples = originals + qa_tripled
    print(f"Total: {len(all_samples)}")

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Write
    output = "data/paper_cpt/v8_augmented.jsonl"
    with open(output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Written to {output}")
    with open(output) as f:
        print(f"Verified: {sum(1 for _ in f)} lines")


if __name__ == "__main__":
    main()
