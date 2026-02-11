"""
Evaluate v3 checkpoints on HELMET HotpotQA (zero-shot, no context).

Usage:
    python scripts/eval_checkpoint_hotpotqa_v3.py --variant 5 --step final
    python scripts/eval_checkpoint_hotpotqa_v3.py --variant 5 --all-steps  # Evaluate all checkpoints
    python scripts/eval_checkpoint_hotpotqa_v3.py --all-variants --all-steps  # Full sweep
"""

import json
import asyncio
import argparse
import re
import string
from collections import Counter
from pathlib import Path

import tinker
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Paths
HELMET_DATA = "/sfs/weka/scratch/ks8vf/exp/HELMET/data/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl"
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RUNS_DIR = "/sfs/weka/scratch/ks8vf/exp/runs"


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def load_helmet_questions(filepath: str, max_samples: int = 300, skip: int = 0) -> list[dict]:
    """Load HELMET HotpotQA questions."""
    with open(filepath) as f:
        data = [json.loads(line) for line in f]
    return data[skip:skip + max_samples]


def get_checkpoint_paths(variant: int) -> dict[str, str]:
    """Get all checkpoint paths for a variant."""
    run_dir = Path(f"{RUNS_DIR}/hotpotqa_v3_{variant}var")
    checkpoints_file = run_dir / "checkpoints.jsonl"

    if not checkpoints_file.exists():
        return {}

    checkpoints = {}
    with open(checkpoints_file) as f:
        for line in f:
            ckpt = json.loads(line)
            checkpoints[ckpt['name']] = ckpt['sampler_path']

    return checkpoints


async def evaluate_checkpoint(
    checkpoint_path: str,
    questions: list[dict],
    max_concurrent: int = 20,
) -> dict:
    """Evaluate a single checkpoint on HotpotQA questions."""

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(model_path=checkpoint_path)

    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    tokenizer = get_tokenizer(MODEL_NAME)
    renderer = get_renderer(renderer_name, tokenizer)

    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=100,
        temperature=0.0,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(sample: dict) -> dict:
        async with semaphore:
            question = sample['question']
            gold_answers = sample.get('answer', sample.get('answers', []))
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            # Zero-shot prompt (matches HELMET no_context format exactly)
            messages = [{"role": "user", "content": f"Write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\nQuestion: {question}"}]

            try:
                response = await completer(messages)
                pred = response['content'].strip()
                # Strip "Answer:" prefix if present
                if pred.lower().startswith("answer:"):
                    pred = pred[7:].strip()
            except Exception as e:
                print(f"Error: {e}")
                pred = ""

            em = max(exact_match(pred, ans) for ans in gold_answers) if gold_answers else 0.0
            f1 = max(f1_score(pred, ans) for ans in gold_answers) if gold_answers else 0.0

            return {"em": em, "f1": f1, "pred": pred, "gold": gold_answers[0] if gold_answers else ""}

    tasks = [process_question(q) for q in questions]
    results = await asyncio.gather(*tasks)

    em_scores = [r['em'] for r in results]
    f1_scores = [r['f1'] for r in results]

    return {
        "em": sum(em_scores) / len(em_scores) * 100,
        "f1": sum(f1_scores) / len(f1_scores) * 100,
        "n_samples": len(questions),
    }


async def eval_split(checkpoint_path: str) -> dict:
    """Evaluate on all three splits: first 60, rest 240, all 300."""
    first60 = load_helmet_questions(HELMET_DATA, max_samples=60, skip=0)
    rest240 = load_helmet_questions(HELMET_DATA, max_samples=240, skip=60)
    all300 = load_helmet_questions(HELMET_DATA, max_samples=300, skip=0)

    r1 = await evaluate_checkpoint(checkpoint_path, first60)
    r2 = await evaluate_checkpoint(checkpoint_path, rest240)
    r3 = await evaluate_checkpoint(checkpoint_path, all300)

    return {
        "first_60": {"em": r1["em"], "f1": r1["f1"]},
        "rest_240": {"em": r2["em"], "f1": r2["f1"]},
        "all_300": {"em": r3["em"], "f1": r3["f1"]},
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, choices=[5, 10, 15], help="Variant count")
    parser.add_argument("--step", type=str, help="Checkpoint step (e.g., '000012', 'final')")
    parser.add_argument("--all-steps", action="store_true", help="Evaluate all checkpoints")
    parser.add_argument("--all-variants", action="store_true", help="Evaluate all variants")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    variants = [5, 10, 15] if args.all_variants else [args.variant]
    results = {}

    for variant in variants:
        if variant is None:
            continue

        checkpoints = get_checkpoint_paths(variant)
        if not checkpoints:
            print(f"No checkpoints found for variant {variant}")
            continue

        steps = list(checkpoints.keys()) if args.all_steps else [args.step]
        results[f"v3_{variant}var"] = {}

        for step in steps:
            if step not in checkpoints:
                print(f"Checkpoint {step} not found for variant {variant}")
                continue

            print(f"\n=== Evaluating v3_{variant}var step {step} ===")
            checkpoint_path = checkpoints[step]

            split_results = await eval_split(checkpoint_path)
            results[f"v3_{variant}var"][step] = split_results

            print(f"  First 60:  EM {split_results['first_60']['em']:.2f}, F1 {split_results['first_60']['f1']:.2f}")
            print(f"  Rest 240:  EM {split_results['rest_240']['em']:.2f}, F1 {split_results['rest_240']['f1']:.2f}")
            print(f"  All 300:   EM {split_results['all_300']['em']:.2f}, F1 {split_results['all_300']['f1']:.2f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
