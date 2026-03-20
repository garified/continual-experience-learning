"""
Evaluate v5 per-slice checkpoints on HELMET HotpotQA (zero-shot, no context).

For each slice, evaluates on:
  - Trained (60 samples): samples[(s-1)*60 : s*60]
  - Untrained (240 samples): remaining samples
  - All 300 samples

Usage:
    python scripts/eval_checkpoint_hotpotqa_v5.py --slice 1 --all-steps
    python scripts/eval_checkpoint_hotpotqa_v5.py --slice 2 --step 000012
    python scripts/eval_checkpoint_hotpotqa_v5.py --slice 1 --base-model
    python scripts/eval_checkpoint_hotpotqa_v5.py --slice 1 --all-steps --output results/v5_s1.json
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


# Slice mapping: each slice's 20 questions map to 60 eval samples (3 depths each)
# Slice s trains on questions [(s-1)*20+1, s*20], which are samples [(s-1)*60, s*60)
SLICE_RANGES = {
    1: (0, 60),
    2: (60, 120),
    3: (120, 180),
    4: (180, 240),
    5: (240, 300),
}


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


def load_helmet_questions(filepath: str, max_samples: int = 300) -> list[dict]:
    """Load all 300 HELMET HotpotQA samples."""
    with open(filepath) as f:
        data = [json.loads(line) for line in f]
    return data[:max_samples]


def get_checkpoint_paths(slice_num: int) -> dict[str, str]:
    """Get all checkpoint paths for a slice."""
    run_dir = Path(f"{RUNS_DIR}/hotpotqa_v5_s{slice_num}")
    checkpoints_file = run_dir / "checkpoints.jsonl"

    if not checkpoints_file.exists():
        return {}

    checkpoints = {}
    with open(checkpoints_file) as f:
        for line in f:
            ckpt = json.loads(line)
            checkpoints[ckpt['name']] = ckpt['sampler_path']

    return checkpoints


def split_samples(all_samples: list[dict], slice_num: int) -> tuple[list[dict], list[dict]]:
    """Split 300 samples into trained (60) and untrained (240) for a given slice."""
    start, end = SLICE_RANGES[slice_num]
    trained = all_samples[start:end]
    untrained = all_samples[:start] + all_samples[end:]
    return trained, untrained


async def evaluate_checkpoint(
    checkpoint_path: str | None,
    questions: list[dict],
    max_concurrent: int = 20,
    use_base_model: bool = False,
) -> dict:
    """Evaluate a single checkpoint (or base model) on HotpotQA questions."""

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
        max_tokens=100,
        temperature=0.0,
    )

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_question(sample: dict, idx: int) -> dict:
        async with semaphore:
            question = sample['question']
            gold_answers = sample.get('answer', sample.get('answers', []))
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            messages = [{"role": "user", "content": f"Write a concise and short answer to the question. Write your answer in the following format:\nAnswer: [answer]\n\nQuestion: {question}"}]

            try:
                response = await completer(messages)
                pred = response['content'].strip()
                if pred.lower().startswith("answer:"):
                    pred = pred[7:].strip()
            except Exception as e:
                print(f"Error on sample {idx}: {e}")
                pred = ""

            em = max(exact_match(pred, ans) for ans in gold_answers) if gold_answers else 0.0
            f1 = max(f1_score(pred, ans) for ans in gold_answers) if gold_answers else 0.0

            return {"em": em, "f1": f1, "pred": pred, "gold": gold_answers[0] if gold_answers else "", "idx": idx}

    tasks = [process_question(q, i) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)

    em_scores = [r['em'] for r in results]
    f1_scores = [r['f1'] for r in results]

    return {
        "em": sum(em_scores) / len(em_scores) * 100 if em_scores else 0,
        "f1": sum(f1_scores) / len(f1_scores) * 100 if f1_scores else 0,
        "n_samples": len(questions),
        "per_sample": results,
    }


async def eval_slice(
    slice_num: int,
    checkpoint_path: str | None = None,
    use_base_model: bool = False,
) -> dict:
    """Evaluate on trained/untrained/all splits for a given slice."""
    all_samples = load_helmet_questions(HELMET_DATA, max_samples=300)
    trained, untrained = split_samples(all_samples, slice_num)

    start, end = SLICE_RANGES[slice_num]
    print(f"  Slice {slice_num}: trained samples [{start}:{end}] ({len(trained)}), untrained ({len(untrained)})")

    # Evaluate all 300 in one batch, then split results
    print(f"  Evaluating all 300 samples...")
    all_result = await evaluate_checkpoint(checkpoint_path, all_samples, use_base_model=use_base_model)

    # Split per-sample results into trained vs untrained
    per_sample = all_result['per_sample']
    trained_results = per_sample[start:end]
    untrained_results = per_sample[:start] + per_sample[end:]

    def compute_metrics(results):
        if not results:
            return {"em": 0, "f1": 0, "n": 0}
        em = sum(r['em'] for r in results) / len(results) * 100
        f1 = sum(r['f1'] for r in results) / len(results) * 100
        return {"em": round(em, 2), "f1": round(f1, 2), "n": len(results)}

    return {
        "trained": compute_metrics(trained_results),
        "untrained": compute_metrics(untrained_results),
        "all_300": compute_metrics(per_sample),
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Slice number (1-5)")
    parser.add_argument("--step", type=str, help="Checkpoint step (e.g., '000012', 'final')")
    parser.add_argument("--all-steps", action="store_true", help="Evaluate all checkpoints")
    parser.add_argument("--base-model", action="store_true", help="Evaluate base model (no training)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    slice_num = args.slice
    results = {}

    if args.base_model:
        print(f"\n=== Evaluating base model on slice {slice_num} splits ===")
        eval_results = await eval_slice(slice_num, use_base_model=True)
        results["base_model"] = eval_results
        print(f"  Trained ({eval_results['trained']['n']}):   EM {eval_results['trained']['em']:.2f}, F1 {eval_results['trained']['f1']:.2f}")
        print(f"  Untrained ({eval_results['untrained']['n']}): EM {eval_results['untrained']['em']:.2f}, F1 {eval_results['untrained']['f1']:.2f}")
        print(f"  All 300:   EM {eval_results['all_300']['em']:.2f}, F1 {eval_results['all_300']['f1']:.2f}")
    else:
        checkpoints = get_checkpoint_paths(slice_num)
        if not checkpoints:
            print(f"No checkpoints found for slice {slice_num}")
            print(f"Expected: {RUNS_DIR}/hotpotqa_v5_s{slice_num}/checkpoints.jsonl")
            return

        steps = list(checkpoints.keys()) if args.all_steps else [args.step]
        results[f"v5_s{slice_num}"] = {}

        for step in steps:
            if step not in checkpoints:
                print(f"Checkpoint {step} not found for slice {slice_num}")
                continue

            print(f"\n=== Evaluating v5_s{slice_num} step {step} ===")
            checkpoint_path = checkpoints[step]

            eval_results = await eval_slice(slice_num, checkpoint_path)
            results[f"v5_s{slice_num}"][step] = eval_results

            print(f"  Trained ({eval_results['trained']['n']}):   EM {eval_results['trained']['em']:.2f}, F1 {eval_results['trained']['f1']:.2f}")
            print(f"  Untrained ({eval_results['untrained']['n']}): EM {eval_results['untrained']['em']:.2f}, F1 {eval_results['untrained']['f1']:.2f}")
            print(f"  All 300:   EM {eval_results['all_300']['em']:.2f}, F1 {eval_results['all_300']['f1']:.2f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.output}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
