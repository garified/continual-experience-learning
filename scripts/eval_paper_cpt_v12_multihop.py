"""
Evaluate paper CPT v12 checkpoints with NOVEL multi-hop questions.

These questions are deliberately different from all 125 training QA pairs.
They require cross-referencing facts from multiple paper sections and
reasoning about relationships not directly stated in any single training sample.

Usage:
    python scripts/eval_paper_cpt_v12_multihop.py --all-steps
    python scripts/eval_paper_cpt_v12_multihop.py --all-steps --output results/paper_cpt_v12_multihop.json
"""

import json
import asyncio
import argparse
from pathlib import Path

import tinker
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RUNS_DIR = "/sfs/weka/scratch/ks8vf/exp/runs/paper_cpt_v12"


# ═══════════════════════════════════════════════════════════════════════════
# MULTI-HOP / CROSS-SECTION QUESTIONS
# Each question requires combining facts from 2+ different paper sections.
# None of these questions appear in the 125 training QA pairs.
# ═══════════════════════════════════════════════════════════════════════════

EVAL_QUESTIONS = [
    # --- Multi-hop: Methodology + Experiments ---
    {
        "id": "mh_1",
        "type": "cross_method_experiment",
        "question": "The ICRL paper describes two instruction strategies (Preset and Autonomous). Which one was used in the Game of 24 experiment, and what was the difference in their success rates after 50 trials?",
        "ground_truth": "Both ICRL Preset and ICRL Autonomous were tested on Game of 24. ICRL Preset achieves 90% success rate after 50 trials. ICRL Preset alternates exploration (even episodes) and exploitation (odd episodes). ICRL Autonomous lets the LLM decide. Both outperform baselines.",
        "key_facts": ["ICRL Preset", "ICRL Autonomous", "90%", "50 trials", "exploration", "exploitation", "even", "odd"],
    },
    {
        "id": "mh_2",
        "type": "cross_method_experiment",
        "question": "In the ICRL paper, the experience buffer stores previous responses and rewards. The ablation study tests what happens with a short context buffer. What specific buffer configuration was tested, and what was the finding compared to the full buffer?",
        "ground_truth": "The ablation tested Short Context where the buffer B is made a deque of length 3 instead of infinite (queue). Only the recent 3 episodes are used in constructing S_0. Performance drops with short context, demonstrating that the growth of context is important for ICRL's improvement.",
        "key_facts": ["deque", "length 3", "3 episodes", "performance drop", "short context"],
    },
    # --- Multi-hop: Reward types across experiments ---
    {
        "id": "mh_3",
        "type": "cross_reward_experiment",
        "question": "The ICRL paper describes sparse and dense reward functions. For each of the three main benchmarks (Game of 24, creative writing, ScienceWorld), which type of reward was used and who provided it?",
        "ground_truth": "Game of 24: dense reward (GPT-4.1 scores each of 4 thinking steps on 0-3 scale, same LLM as policy). Creative writing: sparse reward (only R_T, coherence score from GPT-4.1 comparing against reference, same LLM as policy). ScienceWorld: sparse reward (environment-provided, r and r_* are identical, GPT-4.1 mini as policy).",
        "key_facts": ["dense", "sparse", "GPT-4.1", "0-3 scale", "thinking step", "coherence", "environment-provided", "GPT-4.1 mini"],
    },
    # --- Multi-hop: Theory + Evidence ---
    {
        "id": "mh_4",
        "type": "cross_theory_evidence",
        "question": "The ICRL paper claims the 'reward is enough' hypothesis supports their framework. What specific experimental evidence from the ablation study directly tests whether reward signals are necessary, and what was the result?",
        "ground_truth": "The 'reward is enough' hypothesis states intelligence can be understood as subserving the maximisation of reward. The ablation 'Zero Rewards' sets all rewards to 0 and shows performance drop. The 'Exploration Only without reward signal' method performs significantly worse than ICRL when comparing running max. This demonstrates ICRL generates genuinely better novel responses, not just exploring and picking the best (Best-of-N).",
        "key_facts": ["maximisation of reward", "Zero Rewards", "performance drop", "Exploration Only", "without reward", "significantly worse", "running max", "genuinely better novel responses"],
    },
    # --- Multi-hop: Different models across experiments ---
    {
        "id": "mh_5",
        "type": "cross_model_experiment",
        "question": "Different LLMs are used as the policy across the paper's experiments. Which specific model serves as the policy for Game of 24, which for ScienceWorld, and which open-source models were tested on creative writing and math?",
        "ground_truth": "Game of 24: GPT-4.1 as policy (chosen for excellent long-context capacity). ScienceWorld: GPT-4.1 mini as policy. Open-source models on creative writing and math: Phi-4, Llama-4 Maverick, Qwen3-32B, and Qwen3-32B in thinking mode.",
        "key_facts": ["GPT-4.1", "long-context", "GPT-4.1 mini", "Phi-4", "Llama-4 Maverick", "Qwen3-32B", "thinking mode"],
    },
    # --- Multi-hop: Self-eval + unseen paper abstract ---
    {
        "id": "mh_6",
        "type": "cross_selfreward_evidence",
        "question": "The ICRL paper discusses using the LLM itself as the reward function. In which benchmarks is the reward self-generated by the same LLM, and how does the unseen paper abstract experiment prove that ICRL goes beyond parametric knowledge search?",
        "ground_truth": "Self-generated reward: Game of 24 (GPT-4.1 is both policy and reward) and creative writing (GPT-4.1 scores coherence). The hypothesis is 'evaluation is easier than generation'. The unseen paper abstract experiment uses papers published after the model's training cutoff. Best-of-1024 reaches only 0.44 ROUGE-recall, Self-Refine 0.45, Reflexion 0.46, but ICRL achieves 0.59 over 200 iterations, demonstrating learning from external reward signals beyond parametric knowledge.",
        "key_facts": ["evaluation is easier than generation", "training cutoff", "0.44", "0.45", "0.46", "0.59", "200 iterations", "external reward"],
    },
    # --- Multi-hop: Formal RL definition + ICRL mapping ---
    {
        "id": "mh_7",
        "type": "cross_formalism_practice",
        "question": "The paper formally models LLM token generation as an MDP. What are the state space, action space, and transition function in this formulation, and how does S_0 in ICRL prompting differ from the standard RL initial state?",
        "ground_truth": "State space S = union of V^i for i=1 to infinity (all possible token sequences). Action space A = V (vocabulary, next token). Transition: S_t+1 = [S_t A_t] (concatenation). In standard RL, S_0 is sampled from p_0. In ICRL prompting, S_0 is constructed by concatenating the task description s_task, all previous responses and rewards from the experience buffer, and the ICRL instruction s_ICRL.",
        "key_facts": ["token", "V", "concatenat", "s_task", "experience buffer", "s_ICRL", "previous responses", "rewards"],
    },
    # --- Multi-hop: Context length analysis + ablation ---
    {
        "id": "mh_8",
        "type": "cross_context_ablation",
        "question": "The paper presents both a context length analysis and an ablation on short context. What context lengths were tested in the analysis, on which model, and how do those findings relate to the short context ablation result?",
        "ground_truth": "Context length analysis tested 8k, 16k, and 32k on Qwen3-32B. ICRL surpasses Self-Refine and Reflexion in both Creative Writing and AIME at all context lengths. The short context ablation makes the buffer a deque of length 3. Both findings converge: more context leads to better ICRL performance, and restricting context causes performance drop.",
        "key_facts": ["8k", "16k", "32k", "Qwen3-32B", "Creative Writing", "AIME", "deque", "length 3", "performance drop"],
    },
    # --- Multi-hop: Minimality principle + comparison to search ---
    {
        "id": "mh_9",
        "type": "cross_design_comparison",
        "question": "The paper emphasizes 'minimality' as a key design principle. What specific components does ICRL deliberately exclude, and how does this contrast with the search-based methods (ToT, MCTS, Go-Explore) mentioned in the related work?",
        "ground_truth": "ICRL deliberately excludes textual gradients, prioritized experience replay, sampling-based heuristics, and additional engineered modules. The only supervision is the scalar reward. Search-based methods like Tree-of-Thoughts (ToT), Graph-of-Thoughts (GoT), Monte Carlo Tree Search (MCTS), and Intelligent Go-Explore depend on externally engineered components such as heuristics or memory management, rather than leveraging the model's intrinsic learning ability.",
        "key_facts": ["textual gradients", "prioritized experience replay", "sampling-based heuristics", "scalar reward", "Tree-of-Thoughts", "MCTS", "engineered components", "intrinsic learning"],
    },
    # --- Multi-hop: Contributions + concrete evidence mapping ---
    {
        "id": "mh_10",
        "type": "cross_contribution_evidence",
        "question": "The paper lists three contributions. The second contribution claims 'emergence of RL' supported by five specific observations. Name all five observations and for each, identify which experiment or ablation provides the evidence.",
        "ground_truth": "Five observations: (1) Maximisation of scalar reward signal - shown in Game of 24, creative writing, ScienceWorld return curves. (2) Exploration-exploitation trade-off - shown in Game of 24 oscillations in success rate. (3) Performance improvement from context growth - shown across all benchmarks. (4) Performance drop with short context - shown in ablation (deque of length 3). (5) Performance drop when reward is absent - shown in ablation (Zero Rewards and Exploration Only).",
        "key_facts": ["maximisation", "scalar reward", "exploration-exploitation", "context growth", "short context", "performance drop", "reward is absent", "deque", "Zero Rewards"],
    },
    # --- Multi-hop: Author affiliation + funding + venue ---
    {
        "id": "mh_11",
        "type": "cross_meta_info",
        "question": "Where was the ICRL paper published, what university are the authors from, and what funding sources supported this work?",
        "ground_truth": "Published at ICLR 2026. Authors are from University of Virginia (UVA). Funding: NSF grants III-2128019 and SLES-2331904, Coastal Virginia Center for Cyber Innovation (COVA CCI), Commonwealth Cyber Initiative (CCI), and NSF Grant No. 2124538.",
        "key_facts": ["ICLR 2026", "Virginia", "UVA", "NSF", "III-2128019", "SLES-2331904", "COVA CCI", "Commonwealth Cyber Initiative", "2124538"],
    },
    # --- Multi-hop: Self-Refine collapse + creative writing evidence ---
    {
        "id": "mh_12",
        "type": "cross_failure_evidence",
        "question": "The paper argues that Self-Refine suffers from 'performance collapse' due to hallucinated feedback. Which specific experiment demonstrates this collapse most clearly, and what quantitative evidence supports it?",
        "ground_truth": "Creative writing demonstrates Self-Refine's collapse most clearly. Self-Refine initially matches ICRL in coherence reward, but after extending both methods by 50 additional episodes, ICRL keeps improving while Self-Refine first plateaus then declines, likely due to the significant growth of its context with accumulated hallucinated verbal feedback. Quantitatively, ICRL achieves 86.32% length-controlled win rate against Self-Refine.",
        "key_facts": ["creative writing", "initially matches", "plateaus", "declines", "context", "hallucinated", "verbal feedback", "86.32%"],
    },
    # --- Multi-hop: Prompt optimization vs ICRL ---
    {
        "id": "mh_13",
        "type": "cross_related_distinction",
        "question": "The related work section distinguishes ICRL from prompt optimization methods. What is the key theoretical difference, and how does the 'Exploration Only' ablation result support this distinction?",
        "ground_truth": "Prompt optimization uses numerical scores to guide prompt refinement through top-k selection and error filtering, which is more aligned with in-context supervised learning (filtered behavior cloning) than RL. ICRL enables learning from failure experiences. The 'Exploration Only without reward' ablation performs significantly worse, showing that ICRL's improvement comes not from exploring and picking the best (like prompt optimization / Best-of-N) but from genuinely generating better novel responses through RL.",
        "key_facts": ["prompt optimization", "top-k selection", "supervised learning", "behavior cloning", "failure experiences", "Exploration Only", "significantly worse", "genuinely", "novel responses"],
    },
    # --- Multi-hop: ScienceWorld fairness + compute scaling ---
    {
        "id": "mh_14",
        "type": "cross_fairness_scaling",
        "question": "In the ScienceWorld experiment, the paper gives Reflexion and Self-Refine an advantage that ICRL does not have. What is this advantage, and despite it, how does ICRL compare in terms of test-time compute scaling?",
        "ground_truth": "Reflexion and Self-Refine are allowed access to the reward signals of the current episode before prompting for reflection, unlike ICRL which only uses rewards from previous episodes. Despite this advantage, ICRL outperforms baseline methods by about 20% after enough iterations. ICRL also scales better than baselines not only in terms of number of trials but also in test-time compute budget (in dollar amounts).",
        "key_facts": ["reward signals of the current episode", "unlike ICRL", "fair comparison", "20%", "scales better", "compute budget", "dollar"],
    },
    # --- Multi-hop: Duck test + all five observations ---
    {
        "id": "mh_15",
        "type": "cross_duck_test",
        "question": "The paper uses the phrase 'duck test' to describe their evidence for emergent RL. What does this metaphor mean in context, and which of the five RL-characteristic observations were demonstrated specifically in the Game of 24 experiment?",
        "ground_truth": "The 'duck test' means: if the inference process exhibits all the characteristics expected of an RL algorithm, then it is RL ('if it looks like a duck and quacks like a duck, it is a duck'). In Game of 24: maximisation of scalar reward (success rate increases), exploration-exploitation trade-off (oscillations in ICRL Preset reflect alternating phases), and improvement from context growth (performance increases over trials). Short context and absent reward ablations were also tested on Game of 24.",
        "key_facts": ["duck test", "RL algorithm", "maximisation", "oscillation", "exploration", "exploitation", "alternating", "context grow"],
    },
]


def score_response(response: str, question: dict) -> dict:
    response_lower = response.lower()
    hits = []
    misses = []
    for fact in question["key_facts"]:
        if fact.lower() in response_lower:
            hits.append(fact)
        else:
            misses.append(fact)
    total = len(question["key_facts"])
    score = len(hits) / total if total > 0 else 0.0
    return {"score": round(score, 3), "hits": hits, "misses": misses, "n_hits": len(hits), "n_total": total}


def get_checkpoint_paths() -> dict[str, str]:
    checkpoints_file = Path(RUNS_DIR) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return {}
    checkpoints = {}
    with open(checkpoints_file) as f:
        for line in f:
            ckpt = json.loads(line)
            checkpoints[ckpt['name']] = ckpt['sampler_path']
    return checkpoints


async def evaluate_model(checkpoint_path: str | None, use_base_model: bool = False, max_concurrent: int = 10) -> dict:
    service_client = tinker.ServiceClient()
    if use_base_model:
        sampling_client = await service_client.create_sampling_client_async(base_model=MODEL_NAME)
    else:
        sampling_client = await service_client.create_sampling_client_async(model_path=checkpoint_path)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)

    completer = TinkerMessageCompleter(sampling_client=sampling_client, renderer=renderer, max_tokens=2048, temperature=0.0)
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(q: dict) -> dict:
        async with semaphore:
            messages = [{"role": "user", "content": q["question"]}]
            try:
                response = await completer(messages)
                pred = response['content'].strip()
            except Exception as e:
                print(f"  Error on {q['id']}: {e}")
                pred = ""
            scoring = score_response(pred, q)
            return {"id": q["id"], "type": q["type"], "question": q["question"], "response": pred, "ground_truth": q["ground_truth"], "scoring": scoring}

    tasks = [process_question(q) for q in EVAL_QUESTIONS]
    results = await asyncio.gather(*tasks)

    type_scores = {}
    for r in results:
        t = r["type"]
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(r["scoring"]["score"])

    type_avg = {t: round(sum(s) / len(s), 3) for t, s in type_scores.items()}
    overall = round(sum(r["scoring"]["score"] for r in results) / len(results), 3)
    return {"overall_score": overall, "type_scores": type_avg, "per_question": results}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--base-model", action="store_true")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    all_results = {}

    if args.base_model or args.all_steps:
        print("\n=== Evaluating base model ===")
        base_results = await evaluate_model(None, use_base_model=True)
        all_results["base_model"] = base_results
        print(f"  Overall: {base_results['overall_score']:.3f}")
        for t, s in base_results['type_scores'].items():
            print(f"  {t}: {s:.3f}")

    if not args.base_model:
        checkpoints = get_checkpoint_paths()
        if not checkpoints:
            if not args.base_model:
                print(f"No checkpoints found in {RUNS_DIR}")
                if not all_results:
                    return
        else:
            steps = list(checkpoints.keys()) if args.all_steps else ([args.step] if args.step else [])
            for step in steps:
                if step not in checkpoints:
                    print(f"Checkpoint {step} not found")
                    continue
                print(f"\n=== Evaluating step {step} ===")
                step_results = await evaluate_model(checkpoints[step])
                all_results[f"step_{step}"] = step_results
                print(f"  Overall: {step_results['overall_score']:.3f}")
                for t, s in step_results['type_scores'].items():
                    print(f"  {t}: {s:.3f}")

    if len(all_results) > 1:
        print("\n=== Comparison Summary ===")
        print(f"{'Model':<20} {'Overall':<10} ", end="")
        all_types = sorted(set(t for r in all_results.values() for t in r['type_scores']))
        for t in all_types:
            print(f"{t:<30} ", end="")
        print()
        for name, r in all_results.items():
            print(f"{name:<20} {r['overall_score']:<10.3f} ", end="")
            for t in all_types:
                print(f"{r['type_scores'].get(t, 0):<30.3f} ", end="")
            print()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_results = {}
        for name, r in all_results.items():
            save_results[name] = {
                "overall_score": r["overall_score"],
                "type_scores": r["type_scores"],
                "per_question": [{"id": q["id"], "type": q["type"], "question": q["question"], "response": q["response"], "ground_truth": q["ground_truth"], "score": q["scoring"]["score"], "hits": q["scoring"]["hits"], "misses": q["scoring"]["misses"]} for q in r["per_question"]],
            }
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
