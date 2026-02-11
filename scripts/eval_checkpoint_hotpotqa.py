"""
Evaluate Tinker checkpoints on HELMET HotpotQA (zero-shot, no context).
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
    """Load HELMET HotpotQA questions (dep3 format: 3 depths per question)."""
    with open(filepath) as f:
        data = [json.loads(line) for line in f]
    # Skip first N, then take max_samples
    return data[skip:skip + max_samples]


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
                # Strip "Answer:" prefix if present (model follows HELMET format)
                if pred.lower().startswith("answer:"):
                    pred = pred[7:].strip()
            except Exception as e:
                print(f"Error: {e}")
                pred = ""

            # Compute metrics against all gold answers
            em = max(exact_match(pred, ans) for ans in gold_answers) if gold_answers else 0.0
            f1 = max(f1_score(pred, ans) for ans in gold_answers) if gold_answers else 0.0

            return {"em": em, "f1": f1, "pred": pred, "gold": gold_answers[0] if gold_answers else ""}

    print(f"Evaluating {len(questions)} questions...")
    tasks = [process_question(q) for q in questions]
    results = await asyncio.gather(*tasks)

    em_scores = [r['em'] for r in results]
    f1_scores = [r['f1'] for r in results]

    return {
        "em": sum(em_scores) / len(em_scores) * 100,
        "f1": sum(f1_scores) / len(f1_scores) * 100,
        "n_samples": len(questions),
        "results": results[:5],  # Sample predictions
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Tinker checkpoint path")
    parser.add_argument("--max_samples", type=int, default=300, help="Max questions to evaluate")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N samples")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()

    questions = load_helmet_questions(HELMET_DATA, args.max_samples, args.skip)
    print(f"Loaded {len(questions)} questions (skip={args.skip})")

    print(f"Evaluating checkpoint: {args.checkpoint}")
    metrics = await evaluate_checkpoint(args.checkpoint, questions)

    print(f"\nResults:")
    print(f"  EM: {metrics['em']:.2f}")
    print(f"  F1: {metrics['f1']:.2f}")
    print(f"  N: {metrics['n_samples']}")

    print(f"\nSample predictions:")
    for r in metrics['results']:
        print(f"  Pred: {r['pred'][:50]}... | Gold: {r['gold'][:50]}...")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved to {args.output}")

    return metrics


if __name__ == "__main__":
    asyncio.run(main())
