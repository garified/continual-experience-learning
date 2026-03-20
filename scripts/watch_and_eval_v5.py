"""
Watch for new checkpoints during training and run evals as they appear.

Polls checkpoints.jsonl for new entries, runs eval on each, stops when "final" appears.

Usage:
    python scripts/watch_and_eval_v5.py --slice 1
    python scripts/watch_and_eval_v5.py --slice 2 --poll-interval 60
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Import eval functions from the eval script
from eval_checkpoint_hotpotqa_v5 import eval_slice, get_checkpoint_paths, RUNS_DIR


def get_evaluated_checkpoints(slice_num: int) -> set[str]:
    """Check which checkpoints have already been evaluated (have eval_*.json files)."""
    run_dir = Path(f"{RUNS_DIR}/hotpotqa_v5_s{slice_num}")
    evaluated = set()
    for f in run_dir.glob("eval_*.json"):
        # Extract checkpoint name from eval_000012.json -> 000012
        name = f.stem.replace("eval_", "")
        evaluated.add(name)
    return evaluated


def save_eval_results(slice_num: int, checkpoint_name: str, results: dict):
    """Save evaluation results to a JSON file."""
    run_dir = Path(f"{RUNS_DIR}/hotpotqa_v5_s{slice_num}")
    output_file = run_dir / f"eval_{checkpoint_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {output_file}")


async def watch_and_eval(slice_num: int, poll_interval: int = 30):
    """Watch for checkpoints and evaluate them as they appear."""
    print(f"=== Watching slice {slice_num} for new checkpoints ===")
    print(f"  Poll interval: {poll_interval}s")
    print(f"  Run dir: {RUNS_DIR}/hotpotqa_v5_s{slice_num}")
    print()

    evaluated = get_evaluated_checkpoints(slice_num)
    if evaluated:
        print(f"  Already evaluated: {sorted(evaluated)}")

    while True:
        checkpoints = get_checkpoint_paths(slice_num)

        if not checkpoints:
            print(f"  No checkpoints yet, waiting...")
            time.sleep(poll_interval)
            continue

        # Process checkpoints in order
        for name in sorted(checkpoints.keys(), key=lambda x: (x != "final", x)):
            if name in evaluated:
                continue

            print(f"\n=== Evaluating checkpoint {name} ===")
            checkpoint_path = checkpoints[name]

            try:
                results = await eval_slice(slice_num, checkpoint_path)
                save_eval_results(slice_num, name, results)
                evaluated.add(name)

                print(f"  Trained ({results['trained']['n']}):   EM {results['trained']['em']:.2f}, F1 {results['trained']['f1']:.2f}")
                print(f"  Untrained ({results['untrained']['n']}): EM {results['untrained']['em']:.2f}, F1 {results['untrained']['f1']:.2f}")
                print(f"  All 300:   EM {results['all_300']['em']:.2f}, F1 {results['all_300']['f1']:.2f}")

            except Exception as e:
                print(f"  Error evaluating {name}: {e}")
                continue

            # Check if this was the final checkpoint
            if name == "final":
                print(f"\n=== Training complete, all evals done for slice {slice_num} ===")
                return

        # Wait before polling again
        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Watch and eval checkpoints as they appear")
    parser.add_argument("--slice", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Slice number (1-5)")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between polls (default: 30)")
    args = parser.parse_args()

    asyncio.run(watch_and_eval(args.slice, args.poll_interval))


if __name__ == "__main__":
    main()
