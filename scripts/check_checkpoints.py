"""Check which Tinker checkpoints are still available (weights not deleted).

Usage (from JupyterLab terminal):
    python -u scripts/check_checkpoints.py
"""
import asyncio
import json
from pathlib import Path

import tinker

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RUNS_DIR = "/sfs/weka/scratch/ks8vf/exp/runs"


async def main():
    sc = tinker.ServiceClient()

    # Test base model
    try:
        await asyncio.wait_for(
            sc.create_sampling_client_async(base_model=MODEL_NAME),
            timeout=15,
        )
        print("base_model: OK", flush=True)
    except asyncio.TimeoutError:
        print("base_model: TIMEOUT", flush=True)
    except Exception as e:
        print(f"base_model: FAIL - {e}", flush=True)

    # Test final checkpoint from each run
    runs_dir = Path(RUNS_DIR)
    for run_dir in sorted(runs_dir.glob("hotpotqa_*")):
        ckpt_file = run_dir / "checkpoints.jsonl"
        if not ckpt_file.exists():
            continue

        lines = [json.loads(l) for l in open(ckpt_file) if l.strip()]
        version = run_dir.name.replace("hotpotqa_", "")

        for ckpt in lines:
            label = f"{version}/{ckpt['name']} (ep{ckpt.get('epoch', '?')})"
            try:
                await asyncio.wait_for(
                    sc.create_sampling_client_async(model_path=ckpt["sampler_path"]),
                    timeout=15,
                )
                print(f"{label}: OK", flush=True)
            except asyncio.TimeoutError:
                print(f"{label}: TIMEOUT", flush=True)
            except Exception as e:
                msg = str(e)
                if "404" in msg:
                    print(f"{label}: GONE (weights deleted)", flush=True)
                elif "402" in msg:
                    print(f"{label}: BILLING ERROR", flush=True)
                else:
                    print(f"{label}: FAIL - {msg[:80]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
