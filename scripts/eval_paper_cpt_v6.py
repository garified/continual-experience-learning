"""
Evaluate paper CPT v6 checkpoints: test whether the fine-tuned model
deeply understands the paper 'Reward Is Enough'.

Evaluates both fine-tuned checkpoint(s) and the base model (baseline).

Question types:
  - Broad comprehension
  - Specific facts
  - Methodology
  - Comparisons
  - Ablations
  - Results across benchmarks
  - Verbatim recall
  - Definitions
  - Author/venue

Usage:
    python scripts/eval_paper_cpt_v6.py --all-steps
    python scripts/eval_paper_cpt_v6.py --step 000012
    python scripts/eval_paper_cpt_v6.py --base-model
    python scripts/eval_paper_cpt_v6.py --all-steps --output results/paper_cpt_v6.json
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
RUNS_DIR = "/sfs/weka/scratch/ks8vf/exp/runs/paper_cpt_v6"


# ── Evaluation questions with ground truth ──────────────────────────────

EVAL_QUESTIONS = [
    # Broad comprehension
    {
        "id": "broad_1",
        "type": "broad_comprehension",
        "question": "What are the main contributions of the paper 'Reward Is Enough: LLMs Are In-Context Reinforcement Learners'?",
        "ground_truth": "Three contributions: (1) Introduce the ICRL prompting framework, a minimal design that elicits inference-time self-improvement in LLMs using only scalar rewards. (2) Provide strong evidence suggesting the emergence of RL in LLM's inference time (maximization of scalar reward, exploration-exploitation trade-off, performance improvement from context growth, performance drop with short context, performance drop when reward is absent). (3) Demonstrate significant improvements over Self-Refine and Reflexion across Game of 24, creative writing, ScienceWorld, and Olympiad-level math (AIME and HMMT).",
        "key_facts": ["ICRL prompting framework", "scalar rewards", "inference-time self-improvement", "emergence of RL", "exploration-exploitation", "Game of 24", "creative writing", "ScienceWorld", "AIME", "HMMT", "Self-Refine", "Reflexion"],
    },
    # Specific facts
    {
        "id": "fact_1",
        "type": "specific_fact",
        "question": "What success rate does ICRL Preset achieve on Game of 24 after 50 trials?",
        "ground_truth": "90% success rate, compared to 49% from Best-of-N, 47% from Self-Refine, and 44% from Reflexion.",
        "key_facts": ["90%", "50 trials", "49%", "Best-of-N", "47%", "Self-Refine", "44%", "Reflexion"],
    },
    {
        "id": "fact_2",
        "type": "specific_fact",
        "question": "What model is used as the policy LLM in the Game of 24 experiments in the paper 'Reward Is Enough'?",
        "ground_truth": "GPT-4.1 is used as the policy LLM for Game of 24 experiments.",
        "key_facts": ["GPT-4.1"],
    },
    {
        "id": "fact_3",
        "type": "specific_fact",
        "question": "What length-controlled win rates does ICRL achieve against baselines in the creative writing task?",
        "ground_truth": "59.48% against Reflexion, 78.36% against Long-CoT style prompting, 86.32% against Self-Refine, and 93.81% against Best-of-N.",
        "key_facts": ["59.48%", "78.36%", "86.32%", "93.81%"],
    },
    # Methodology
    {
        "id": "method_1",
        "type": "methodology",
        "question": "How does the experience buffer work in ICRL prompting as described in the paper 'Reward Is Enough'?",
        "ground_truth": "An experience buffer B stores the LLM's responses and rewards for the task in previous episodes. Previous attempts and rewards are concatenated as many as the context window allows. The LLM is expected to reinforcement learn from the experiences in context during inference time. The hypothesis is that the pretrained LLM already has ICRL ability, and the buffer activates it.",
        "key_facts": ["experience buffer", "stores responses and rewards", "previous episodes", "concatenated", "context window", "pretrained LLM", "innate ICRL ability"],
    },
    {
        "id": "method_2",
        "type": "methodology",
        "question": "What are the two ICRL instruction strategies described in the paper 'Reward Is Enough'?",
        "ground_truth": "Two strategies: (1) ICRL Preset: alternates between exploration and exploitation instructions - even episodes use exploration, odd episodes use exploitation. (2) ICRL Autonomous: always provides the 'exploration or exploitation' instruction and lets the LLM decide which to use.",
        "key_facts": ["ICRL Preset", "ICRL Autonomous", "alternates", "exploration", "exploitation", "even episodes", "odd episodes", "LLM decide"],
    },
    # Comparisons
    {
        "id": "compare_1",
        "type": "comparison",
        "question": "How does ICRL prompting differ from Self-Refine and Reflexion according to the paper 'Reward Is Enough'?",
        "ground_truth": "ICRL prompting uses scalar reward signals directly without any verbal feedback. Self-Refine asks the LLM to provide textual verbal feedback without a reward function. Reflexion generates reflection according to the reward. The comparison is essentially scalar feedback vs verbal feedback. Self-revision methods are prone to hallucinated feedback that accumulates, leading to performance collapse. ICRL requires only numerical rewards without prescribing new instructions.",
        "key_facts": ["scalar reward", "no verbal feedback", "Self-Refine verbal feedback", "Reflexion reflection", "hallucinated feedback", "performance collapse"],
    },
    # Ablations
    {
        "id": "ablation_1",
        "type": "ablation",
        "question": "What happens when rewards are set to zero in the ICRL ablation study?",
        "ground_truth": "Setting all rewards to 0 leads to performance drop. The ablation study shows that reward signals are important for ICRL's performance. The 'exploration only without reward signal' method performs significantly worse than the full approach when comparing maximum performance over time (running max).",
        "key_facts": ["performance drop", "rewards set to 0", "exploration only without reward", "significantly worse", "running max"],
    },
    {
        "id": "ablation_2",
        "type": "ablation",
        "question": "What ablations were tested in the paper 'Reward Is Enough' and what were the findings?",
        "ground_truth": "Five ablations: (1) Zero Rewards: all rewards set to 0 - performance drops. (2) Short Context: buffer is a deque of length 3 instead of infinite - performance drops. (3) Exploration Only: just asks for different responses without reward - performs significantly worse. (4) Exploitation Only: always uses exploitation instruction with reward - performs well. (5) No ICRL Instruction: entirely removes s_ICRL. Both full ICRL methods and exploitation-only with reward perform best, showing robustness. Key finding: improvement is not just from exploring and picking best (Best-of-N), but ICRL generates genuinely better novel responses.",
        "key_facts": ["Zero Rewards", "Short Context", "deque of length 3", "Exploration Only", "Exploitation Only", "No ICRL Instruction", "performance drop", "robustness", "genuinely better novel responses"],
    },
    # Results across benchmarks
    {
        "id": "results_1",
        "type": "results",
        "question": "What are the results of ICRL on ScienceWorld as described in the paper 'Reward Is Enough'?",
        "ground_truth": "ICRL prompting outperforms baseline methods by about 20% after enough iterations on ScienceWorld. ScienceWorld is an interactive text-based benchmark with 30 science-experiment tasks. GPT-4.1 mini is used as the policy. The environment provides sparse rewards. ICRL also scales better than baselines in terms of test-time compute budget (in dollar amounts).",
        "key_facts": ["outperforms by about 20%", "30 science-experiment tasks", "GPT-4.1 mini", "sparse rewards", "scales better", "test-time compute budget"],
    },
    {
        "id": "results_2",
        "type": "results",
        "question": "What ROUGE-recall score does ICRL achieve on the unseen paper abstract generation task, and how does it compare to baselines?",
        "ground_truth": "ICRL achieves 0.59 ROUGE-recall over 200 iterations. Best-of-1024 reaches only 0.44. Self-Refine plateaus at 0.45. Reflexion reaches 0.46. This demonstrates ICRL can learn from external reward signals and is not limited by the model's pre-training knowledge.",
        "key_facts": ["0.59", "200 iterations", "0.44", "Best-of-1024", "0.45", "Self-Refine", "0.46", "Reflexion", "external reward signal"],
    },
    # Verbatim recall
    {
        "id": "verbatim_1",
        "type": "verbatim_recall",
        "question": "Reproduce the abstract of 'Reward Is Enough: LLMs Are In-Context Reinforcement Learners' as closely as possible.",
        "ground_truth": "Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.",
        "key_facts": ["sequential decision-making", "emerges during inference time", "in-context RL", "ICRL prompting", "numerical scalar feedback", "concatenates all prior responses", "response quality improves", "Game of 24", "creative writing", "ScienceWorld", "AIME", "HMMT", "Self-Refine", "Reflexion", "test-time scaling"],
    },
    # Definitions
    {
        "id": "def_1",
        "type": "definition",
        "question": "What is in-context reinforcement learning (ICRL) as defined in the paper 'Reward Is Enough'?",
        "ground_truth": "ICRL is an emerging inference-time compute paradigm where the RL process occurs in the inference time (the forward pass) of the network without any parameter update. In ICRL, the policy is additionally conditioned on a context C_t. After pretraining on a wide range of tasks, the parameter theta_* is kept fixed. The quality of actions improves as the context grows in new tasks. Since parameters are fixed, improvement comes only from the increase of context. This is called in-context policy improvement. This improvement is observed even on out-of-distribution tasks.",
        "key_facts": ["inference-time", "forward pass", "no parameter update", "context C_t", "theta_* fixed", "in-context policy improvement", "out-of-distribution"],
    },
    {
        "id": "def_2",
        "type": "definition",
        "question": "What is the 'reward is enough' hypothesis referenced in the paper?",
        "ground_truth": "The 'reward is enough' hypothesis states that 'intelligence, and its associated abilities, can be understood as subserving the maximisation of reward.' The paper also references the reward hypothesis: 'all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward).'",
        "key_facts": ["intelligence", "maximisation of reward", "reward hypothesis", "cumulative sum", "scalar signal"],
    },
    # Author/venue
    {
        "id": "author_1",
        "type": "author_venue",
        "question": "Who wrote 'Reward Is Enough: LLMs Are In-Context Reinforcement Learners' and where was it published?",
        "ground_truth": "Authors: Kefan Song, Amir Moeini, Peng Wang, Lei Gong, Rohan Chandra, Shangtong Zhang, Yanjun Qi (University of Virginia / UVA). Published at ICLR 2026.",
        "key_facts": ["Kefan Song", "Amir Moeini", "Peng Wang", "Lei Gong", "Rohan Chandra", "Shangtong Zhang", "Yanjun Qi", "UVA", "ICLR 2026"],
    },
    # Additional methodology
    {
        "id": "method_3",
        "type": "methodology",
        "question": "What types of reward functions does ICRL prompting support according to the paper?",
        "ground_truth": "ICRL supports sparse rewards (outcome reward model, where r(s) is nonzero only at terminal states) and dense rewards (progress reward model, where r(s) can be nonzero for non-terminal states). The reward function can be rule-based, learned separately, or instantiated via the same LLM for self-evaluation. When the LLM self-evaluates, there is no external feedback at all, yet performance still improves due to the hypothesis that evaluation is easier than generation.",
        "key_facts": ["sparse rewards", "dense rewards", "rule-based", "learned separately", "LLM self-evaluation", "no external feedback", "evaluation easier than generation"],
    },
    # Additional results
    {
        "id": "results_3",
        "type": "results",
        "question": "What open-source models were tested with ICRL in the paper 'Reward Is Enough', and on what tasks?",
        "ground_truth": "Open-source models tested include Phi-4, Llama-4 Maverick, Qwen3-32B, and Qwen3-32B in thinking mode, tested on creative writing and Olympiad-level math (AIME, HMMT). ICRL consistently outperforms baselines like Self-Refine and Reflexion in all settings, with improvements of up to 10-20 points over the base model.",
        "key_facts": ["Phi-4", "Llama-4 Maverick", "Qwen3-32B", "thinking mode", "creative writing", "AIME", "HMMT", "10-20 points"],
    },
]


def score_response(response: str, question: dict) -> dict:
    """Score a response by checking which key facts are present."""
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

    return {
        "score": round(score, 3),
        "hits": hits,
        "misses": misses,
        "n_hits": len(hits),
        "n_total": total,
    }


def get_checkpoint_paths() -> dict[str, str]:
    """Get all checkpoint paths for paper_cpt_v6."""
    checkpoints_file = Path(RUNS_DIR) / "checkpoints.jsonl"
    if not checkpoints_file.exists():
        return {}

    checkpoints = {}
    with open(checkpoints_file) as f:
        for line in f:
            ckpt = json.loads(line)
            checkpoints[ckpt['name']] = ckpt['sampler_path']
    return checkpoints


async def evaluate_model(
    checkpoint_path: str | None,
    use_base_model: bool = False,
    max_concurrent: int = 10,
) -> dict:
    """Evaluate a model on all paper comprehension questions."""

    service_client = tinker.ServiceClient()
    if use_base_model:
        sampling_client = await service_client.create_sampling_client_async(base_model=MODEL_NAME)
    else:
        sampling_client = await service_client.create_sampling_client_async(model_path=checkpoint_path)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=2048,
        temperature=0.0,
    )

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

            return {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "response": pred,
                "ground_truth": q["ground_truth"],
                "scoring": scoring,
            }

    tasks = [process_question(q) for q in EVAL_QUESTIONS]
    results = await asyncio.gather(*tasks)

    # Aggregate by type
    type_scores = {}
    for r in results:
        t = r["type"]
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(r["scoring"]["score"])

    type_avg = {t: round(sum(s) / len(s), 3) for t, s in type_scores.items()}
    overall = round(sum(r["scoring"]["score"] for r in results) / len(results), 3)

    return {
        "overall_score": overall,
        "type_scores": type_avg,
        "per_question": results,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, help="Checkpoint step (e.g., '000012', 'final')")
    parser.add_argument("--all-steps", action="store_true", help="Evaluate all checkpoints")
    parser.add_argument("--base-model", action="store_true", help="Evaluate base model (no training)")
    parser.add_argument("--output", type=str, help="Output JSON file")
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
                print(f"Expected: {RUNS_DIR}/checkpoints.jsonl")
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

    # Print comparison summary
    if len(all_results) > 1:
        print("\n=== Comparison Summary ===")
        print(f"{'Model':<20} {'Overall':<10} ", end="")
        all_types = set()
        for r in all_results.values():
            all_types.update(r['type_scores'].keys())
        all_types = sorted(all_types)
        for t in all_types:
            print(f"{t:<20} ", end="")
        print()
        for name, r in all_results.items():
            print(f"{name:<20} {r['overall_score']:<10.3f} ", end="")
            for t in all_types:
                print(f"{r['type_scores'].get(t, 0):<20.3f} ", end="")
            print()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        # Remove full responses for readability, keep summary
        save_results = {}
        for name, r in all_results.items():
            save_results[name] = {
                "overall_score": r["overall_score"],
                "type_scores": r["type_scores"],
                "per_question": [
                    {
                        "id": q["id"],
                        "type": q["type"],
                        "question": q["question"],
                        "response": q["response"],
                        "ground_truth": q["ground_truth"],
                        "score": q["scoring"]["score"],
                        "hits": q["scoring"]["hits"],
                        "misses": q["scoring"]["misses"],
                    }
                    for q in r["per_question"]
                ],
            }
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())
