"""
Prepare v11 training data: targeted QA for methodology, ablation, definition.

v10 analysis: author_venue perfect (1.0), specific_fact strong (0.667),
but methodology (0.191), ablation (0.100), definition (0.071) still weak.

Root cause: model fails to recall exact paper terminology
  - Definitions: misses "forward pass", "no parameter update", "theta_* fixed",
    "in-context policy improvement", "out-of-distribution", "intelligence",
    "maximisation of reward", "cumulative sum"
  - Ablations: hallucinates wrong names, misses "Zero Rewards", "Short Context",
    "Exploration Only", "Exploitation Only", "No ICRL Instruction", "running max"
  - Methodology: confuses ICRL Preset/Autonomous with baselines, misses
    "even episodes", "odd episodes", "LLM decide", "innate ICRL ability",
    "evaluation easier than generation"

v11 strategy:
  - All v10 QA (85) + ~45 new QA targeting exact key_fact terms
  - QA × 10 copies (up from 8) for more emphasis
  - Keep paraphrases
  - Same LR=5e-5, LoRA=64, 4 epochs

Output: data/paper_cpt/v11_augmented.jsonl
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


def generate_v11_qa():
    """Generate QA pairs targeting methodology, ablation, and definition weaknesses."""
    qa = []

    # ═══════════════════════════════════════════════
    # DEFINITION: "What is ICRL?" (def_1)
    # Key facts needed: inference-time, forward pass, no parameter update,
    #   context C_t, theta_* fixed, in-context policy improvement, out-of-distribution
    # ═══════════════════════════════════════════════

    qa.append(s(
        "Define in-context reinforcement learning (ICRL) precisely as described in the paper.",
        f"In-context reinforcement learning (ICRL), as defined in '{PAPER_TITLE}', is an emerging inference-time compute paradigm where the RL process occurs in the forward pass of the network without any parameter update. The policy is conditioned on a context C_t. After pretraining, the parameter theta_* is kept fixed. Action quality improves as the context grows — this is called in-context policy improvement. Crucially, this improvement is observed even on out-of-distribution tasks."
    ))
    qa.append(s(
        "What is ICRL according to the paper 'Reward Is Enough'?",
        f"ICRL (in-context RL) is an inference-time compute paradigm. The RL process occurs in the forward pass without any parameter update. theta_* is fixed after pretraining. The policy is conditioned on context C_t. As context grows, in-context policy improvement occurs. This works even on out-of-distribution tasks."
    ))
    qa.append(s(
        "Explain the formal definition of ICRL from the paper.",
        f"ICRL is defined as an inference-time paradigm where RL happens during the forward pass with no parameter update. Parameters theta_* remain fixed. The policy is conditioned on context C_t that accumulates past experience. The key property is in-context policy improvement: quality improves as context grows, even on out-of-distribution tasks not seen during pretraining."
    ))
    qa.append(s(
        "Does ICRL update model parameters?",
        f"No. In ICRL, there is no parameter update. The parameter theta_* is kept fixed after pretraining. All improvement comes from the growing context C_t during the forward pass. This is the defining characteristic of ICRL as an inference-time compute paradigm. The improvement is called in-context policy improvement and works even on out-of-distribution tasks."
    ))
    qa.append(s(
        "What is in-context policy improvement?",
        f"In-context policy improvement is the phenomenon where action quality improves as the context C_t grows, with theta_* fixed and no parameter update. It happens during inference-time (the forward pass). This improvement is observed even on out-of-distribution tasks. It is the core mechanism behind ICRL."
    ))

    # ═══════════════════════════════════════════════
    # DEFINITION: "reward is enough" hypothesis (def_2)
    # Key facts: intelligence, maximisation of reward, reward hypothesis,
    #   cumulative sum, scalar signal
    # ═══════════════════════════════════════════════

    qa.append(s(
        "What is the 'reward is enough' hypothesis?",
        f"The 'reward is enough' hypothesis states that 'intelligence, and its associated abilities, can be understood as subserving the maximisation of reward.' The related reward hypothesis states: 'all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward).' ICRL prompting complies with both hypotheses by using only a scalar signal as feedback."
    ))
    qa.append(s(
        "What philosophical hypotheses does the ICRL paper reference?",
        f"The paper references two hypotheses: (1) The 'reward is enough' hypothesis: 'intelligence, and its associated abilities, can be understood as subserving the maximisation of reward.' (2) The reward hypothesis: 'all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward).' ICRL prompting is designed to comply with both."
    ))
    qa.append(s(
        "What does 'reward is enough' mean in the context of the ICRL paper?",
        f"'Reward is enough' references the hypothesis that intelligence can be understood as subserving the maximisation of reward. The reward hypothesis further states that goals and purposes can be thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward). The ICRL paper demonstrates this by showing LLMs can self-improve using only scalar reward signals."
    ))
    qa.append(s(
        "What is the reward hypothesis?",
        f"The reward hypothesis states: 'all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward).' The related 'reward is enough' hypothesis states that intelligence can be understood as subserving the maximisation of reward. The ICRL paper uses only scalar signal feedback, consistent with both hypotheses."
    ))

    # ═══════════════════════════════════════════════
    # METHODOLOGY: ICRL Preset vs Autonomous (method_2)
    # Key facts: ICRL Preset, ICRL Autonomous, alternates,
    #   exploration, exploitation, even episodes, odd episodes, LLM decide
    # ═══════════════════════════════════════════════

    qa.append(s(
        "What are the two ICRL instruction strategies?",
        f"The two strategies are: (1) ICRL Preset — alternates between exploration and exploitation instructions. Even episodes use exploration (try something different). Odd episodes use exploitation (produce the best response). (2) ICRL Autonomous — always provides 'exploration or exploitation' and lets the LLM decide which to use."
    ))
    qa.append(s(
        "Describe ICRL Preset in detail.",
        f"ICRL Preset alternates between exploration and exploitation instructions on a fixed schedule. Even episodes use the exploration instruction (asking the model to diverge from previous attempts). Odd episodes use the exploitation instruction (asking the model to produce the best response based on high-reward past attempts). This is one of two ICRL instruction strategies, the other being ICRL Autonomous."
    ))
    qa.append(s(
        "Describe ICRL Autonomous in detail.",
        f"ICRL Autonomous always provides the combined 'exploration or exploitation' instruction and lets the LLM decide which strategy to use at each episode. Unlike ICRL Preset which alternates on a fixed schedule (even episodes exploration, odd episodes exploitation), ICRL Autonomous gives the model the agency to choose."
    ))
    qa.append(s(
        "How does ICRL Preset differ from ICRL Autonomous?",
        f"ICRL Preset alternates exploration and exploitation on a fixed schedule: even episodes use exploration, odd episodes use exploitation. ICRL Autonomous always gives both options and lets the LLM decide which to use. Both achieve strong performance, but ICRL Preset follows a predetermined pattern while ICRL Autonomous gives the LLM agency."
    ))
    qa.append(s(
        "When does ICRL Preset use exploration vs exploitation?",
        f"ICRL Preset alternates on a fixed schedule: even episodes use the exploration instruction (try something different), odd episodes use the exploitation instruction (produce the best response). This is different from ICRL Autonomous where the LLM decide which to use."
    ))

    # ═══════════════════════════════════════════════
    # METHODOLOGY: Experience buffer (method_1)
    # Key facts: experience buffer, stores responses and rewards,
    #   previous episodes, concatenated, context window,
    #   pretrained LLM, innate ICRL ability
    # ═══════════════════════════════════════════════

    qa.append(s(
        "How does the experience buffer work in ICRL?",
        f"The experience buffer B stores responses and rewards from previous episodes. These are concatenated into the prompt, filling as much of the context window as possible. The hypothesis is that the pretrained LLM already has an innate ICRL ability — the buffer activates this latent capability by providing in-context experience for the model to learn from."
    ))
    qa.append(s(
        "What is stored in the ICRL experience buffer?",
        f"The experience buffer stores responses and rewards from previous episodes. These stored experiences are concatenated into the initial prompt S_0, using as much of the context window as allowed. The key hypothesis is that the pretrained LLM already has innate ICRL ability, and the experience buffer activates it."
    ))
    qa.append(s(
        "Why does ICRL use an experience buffer?",
        f"The experience buffer stores responses and rewards from previous episodes and concatenates them into context. The hypothesis is that the pretrained LLM already has innate ICRL ability. The buffer activates this by filling the context window with past attempts and their rewards, allowing the model to learn from experience during inference."
    ))

    # ═══════════════════════════════════════════════
    # METHODOLOGY: Reward types (method_3)
    # Key facts: sparse rewards, dense rewards, rule-based,
    #   learned separately, LLM self-evaluation,
    #   no external feedback, evaluation easier than generation
    # ═══════════════════════════════════════════════

    qa.append(s(
        "What types of rewards does ICRL support?",
        f"ICRL supports two reward types: (1) sparse rewards — only the terminal reward R_T is nonzero, (2) dense rewards — intermediate rewards can also be nonzero. The reward function can be rule-based, learned separately, or via LLM self-evaluation. When the LLM does self-evaluation, there is no external feedback at all. Performance still improves because evaluation is easier than generation."
    ))
    qa.append(s(
        "Can ICRL work without external feedback?",
        f"Yes. When the reward is instantiated via LLM self-evaluation, there is no external feedback at all. The same LLM serves as both policy and reward model. Performance still improves because evaluation is easier than generation. The reward can also be rule-based or learned separately (external feedback)."
    ))
    qa.append(s(
        "What reward functions does ICRL use?",
        f"ICRL supports three types of reward functions: (1) rule-based rewards, (2) learned separately (external), and (3) LLM self-evaluation (no external feedback). It also distinguishes sparse rewards (only terminal R_T nonzero) from dense rewards (intermediate rewards can be nonzero). The hypothesis that evaluation is easier than generation explains why self-eval works."
    ))
    qa.append(s(
        "Why does LLM self-evaluation work as a reward in ICRL?",
        f"LLM self-evaluation works as a reward function because evaluation is easier than generation. Even with no external feedback (the LLM is both policy and reward model), ICRL still improves. This is because the LLM can more accurately judge response quality than generate the optimal response, providing meaningful sparse rewards or dense rewards to guide learning."
    ))

    # ═══════════════════════════════════════════════
    # ABLATION: All five ablations (ablation_2)
    # Key facts: Zero Rewards, Short Context, deque of length 3,
    #   Exploration Only, Exploitation Only, No ICRL Instruction,
    #   performance drop, robustness, genuinely better novel responses
    # ═══════════════════════════════════════════════

    qa.append(s(
        "Name and describe all five ablations in the ICRL paper.",
        f"The five ablations are: (1) Zero Rewards — all rewards set to 0, causes performance drop. (2) Short Context — buffer is a deque of length 3 (only 3 most recent episodes), causes performance drop. (3) Exploration Only — no reward signal, model just explores, performs significantly worse. (4) Exploitation Only — always exploitation instruction with reward, performs well. (5) No ICRL Instruction — s_ICRL removed entirely. Both full ICRL and Exploitation Only with reward show robustness. Key finding: ICRL generates genuinely better novel responses, not just Best-of-N."
    ))
    qa.append(s(
        "What are the exact names of the five ICRL ablation conditions?",
        f"The exact names are: Zero Rewards, Short Context, Exploration Only, Exploitation Only, and No ICRL Instruction. In Zero Rewards, all rewards are set to 0. In Short Context, the buffer is a deque of length 3. In Exploration Only, no reward signal is provided. In Exploitation Only, the exploitation instruction is always used with reward. In No ICRL Instruction, s_ICRL is removed. The methods with reward show robustness and generate genuinely better novel responses."
    ))
    qa.append(s(
        "What did the ablation study prove about ICRL?",
        f"The ablation study proved three things: (1) Reward signals matter — Zero Rewards causes performance drop. (2) Context length matters — Short Context (deque of length 3) causes performance drop. (3) ICRL generates genuinely better novel responses, not just Best-of-N — the Exploration Only condition without reward performs significantly worse than full ICRL. The system shows robustness across Exploitation Only and No ICRL Instruction variants."
    ))
    qa.append(s(
        "What is the Short Context ablation?",
        f"In the Short Context ablation, the experience buffer is constrained to a deque of length 3 (only the 3 most recent episodes instead of the full history). This causes performance drop, proving that more context leads to better ICRL performance. The normal ICRL uses no upper bound on buffer length."
    ))
    qa.append(s(
        "What is the Zero Rewards ablation?",
        f"In the Zero Rewards ablation, all reward signals are set to 0, providing no informative feedback. This causes performance drop, proving that reward signals are essential for ICRL. The Exploration Only ablation (no reward) performs even worse, and the running max shows these variants substantially underperform full ICRL."
    ))

    # ═══════════════════════════════════════════════
    # ABLATION: Zero rewards specific (ablation_1)
    # Key facts: performance drop, rewards set to 0,
    #   exploration only without reward, significantly worse, running max
    # ═══════════════════════════════════════════════

    qa.append(s(
        "What happens when all rewards are set to 0 in ICRL?",
        f"When rewards are set to 0 in the Zero Rewards ablation, there is a clear performance drop. The exploration only without reward condition performs even worse — significantly worse than full ICRL. This is clearly visible in the running max metric. These findings prove that reward signals are critical for ICRL's improvements."
    ))
    qa.append(s(
        "Compare zero rewards vs exploration only in the ICRL ablation.",
        f"Both cause performance drop, but exploration only without reward is significantly worse than Zero Rewards. The running max shows that exploration-only (green curve) achieves much lower performance than full ICRL. This proves ICRL generates genuinely better novel responses rather than just selecting the best from random exploration."
    ))

    # ═══════════════════════════════════════════════
    # COMPARISON: ICRL vs Self-Refine vs Reflexion (compare_1)
    # Key facts: scalar reward, no verbal feedback,
    #   Self-Refine verbal feedback, Reflexion reflection,
    #   hallucinated feedback, performance collapse
    # ═══════════════════════════════════════════════

    qa.append(s(
        "How does ICRL differ from Self-Refine?",
        f"ICRL uses only scalar reward signals with no verbal feedback. Self-Refine uses verbal feedback — it asks the LLM to provide textual self-critique. Self-Refine's verbal feedback is prone to hallucinated feedback that accumulates across iterations, eventually causing performance collapse. ICRL avoids this by using only numerical rewards."
    ))
    qa.append(s(
        "How does ICRL differ from Reflexion?",
        f"ICRL uses only scalar reward with no verbal feedback. Reflexion generates textual reflection based on the reward signal. Like Self-Refine, Reflexion's verbal feedback can include hallucinated feedback that accumulates, leading to performance collapse over many iterations. ICRL's scalar reward approach avoids this degradation."
    ))
    qa.append(s(
        "Why do verbal feedback methods fail compared to ICRL?",
        f"Self-Refine and Reflexion use verbal feedback which is prone to hallucinated feedback that accumulates across iterations. This leads to performance collapse. ICRL uses only scalar reward with no verbal feedback, providing a clean signal that doesn't degrade. The scalar reward approach is the key differentiator."
    ))

    # ═══════════════════════════════════════════════
    # Additional broad comprehension QA
    # (already strong but reinforce)
    # ═══════════════════════════════════════════════

    qa.append(s(
        f"Explain '{PAPER_TITLE}' in detail.",
        f"'{PAPER_TITLE}' (ArXiv 2506.06303, ICLR 2026) by {AUTHORS} (University of Virginia (UVA)) demonstrates that reinforcement learning emerges during LLM inference time, termed in-context RL (ICRL).\n\n**Core Framework — ICRL Prompting**: A multi-round prompting approach where after each response, the model receives a numerical scalar reward. The context concatenates all prior responses and rewards. Response quality improves as context grows. The design is deliberately minimal — only scalar rewards, no verbal feedback.\n\n**Components**: (1) LLM as policy π_θ, (2) Scalar reward function (sparse or dense, can be rule-based, learned separately, or LLM self-evaluation — when using self-evaluation there is no external feedback), (3) Experience buffer storing responses and rewards from previous episodes, (4) ICRL instructions with two strategies: ICRL Preset (alternates exploration on even episodes and exploitation on odd episodes) and ICRL Autonomous (lets the LLM decide).\n\n**ICRL Definition**: An inference-time paradigm where RL occurs in the forward pass with no parameter update. theta_* is fixed. Policy is conditioned on context C_t. In-context policy improvement occurs even on out-of-distribution tasks.\n\n**Experiments**: Game of 24 with GPT-4.1 — 90% success vs 49% Best-of-N, 47% Self-Refine, 44% Reflexion after 50 trials. Creative writing — win rates of 59.48% vs Reflexion, 78.36% vs Long-CoT, 86.32% vs Self-Refine, 93.81% vs Best-of-N. ScienceWorld with GPT-4.1 mini — 20% improvement. Olympiad math with Phi-4, Llama-4 Maverick, Qwen3-32B — 10-20 point gains. Unseen abstracts — 0.59 ROUGE-recall vs 0.44-0.46 baselines.\n\n**Ablations**: Five conditions — Zero Rewards (performance drop), Short Context with deque of length 3 (performance drop), Exploration Only without reward (significantly worse), Exploitation Only (performs well), No ICRL Instruction. Key finding: ICRL generates genuinely better novel responses, not just Best-of-N. System shows robustness.\n\n**vs Baselines**: Self-Refine uses verbal feedback prone to hallucinated feedback that accumulates, causing performance collapse. Reflexion generates reflection from reward. ICRL uses only scalar reward with no verbal feedback.\n\n**Hypotheses**: References the 'reward is enough' hypothesis — intelligence can be understood as subserving the maximisation of reward. The reward hypothesis — goals are maximization of cumulative sum of scalar signal."
    ))

    qa.append(s(
        "What is the complete methodology of ICRL prompting?",
        f"ICRL prompting has four components: (1) The LLM as policy π_θ. (2) A scalar reward function — either sparse rewards (only terminal R_T nonzero) or dense rewards. Can be rule-based, learned separately, or via LLM self-evaluation (no external feedback, works because evaluation is easier than generation). (3) An experience buffer that stores responses and rewards from previous episodes, concatenated into the prompt using as much context window as possible. The pretrained LLM has innate ICRL ability that the buffer activates. (4) ICRL instructions with two strategies: ICRL Preset (alternates exploration on even episodes, exploitation on odd episodes) and ICRL Autonomous (lets the LLM decide). The model receives only scalar reward, no verbal feedback."
    ))

    return qa


def main():
    random.seed(42)

    # Load all unique QA from v10
    qa_v10 = []
    seen = set()
    with open("data/paper_cpt/v10_augmented.jsonl") as f:
        for line in f:
            d = json.loads(line)
            q = d["messages"][0]["content"]
            if q != "." and q not in seen:
                seen.add(q)
                qa_v10.append(d)
    print(f"v10 QA pairs (unique): {len(qa_v10)}")

    # Generate v11-specific QA
    v11_qa = generate_v11_qa()
    print(f"v11 new QA pairs: {len(v11_qa)}")

    # Combine all unique QA
    all_qa = qa_v10.copy()
    for q in v11_qa:
        user_msg = q["messages"][0]["content"]
        if user_msg not in seen:
            seen.add(user_msg)
            all_qa.append(q)
        else:
            # Update existing QA with v11 version (better answers)
            for i, existing in enumerate(all_qa):
                if existing["messages"][0]["content"] == user_msg:
                    all_qa[i] = q
                    break
    print(f"Total unique QA: {len(all_qa)}")

    # QA × 10 copies
    qa_repeated = all_qa * 10
    print(f"QA × 10: {len(qa_repeated)}")

    # Load paraphrases
    paraphrases = []
    with open("data/paper_cpt/v6_augmented.jsonl") as f:
        for line in f:
            paraphrases.append(json.loads(line))
    print(f"Paraphrases: {len(paraphrases)}")

    # Combine
    all_samples = qa_repeated + paraphrases
    print(f"Total: {len(all_samples)}")

    # Shuffle
    random.seed(42)
    random.shuffle(all_samples)

    # Write
    output = "data/paper_cpt/v11_augmented.jsonl"
    with open(output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Written to {output}")
    with open(output) as f:
        print(f"Verified: {sum(1 for _ in f)} lines")


if __name__ == "__main__":
    main()
