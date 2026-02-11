"""
SFT training on HotpotQA data v4: All 100 questions with Type1+Type2.

Experiments:
- v4a: LR = 1e-5 (same as v2)
- v4b: LR = 5e-6 (half of v2)

Usage:
    python scripts/train_hotpotqa_sft_v4.py --lr 1e-5   # v4a
    python scripts/train_hotpotqa_sft_v4.py --lr 5e-6   # v4b
"""

import chz
import sys
import asyncio
import argparse
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA_FILE = "/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_100/combined_train.jsonl"

# Conservative hyperparameters (same as v2)
BATCH_SIZE = 32
LORA_RANK = 64
NUM_EPOCHS = 4
SAVE_EVERY = 12


def get_log_path(lr: float) -> str:
    """Get log path based on LR."""
    if lr == 1e-5:
        return "/sfs/weka/scratch/ks8vf/exp/runs/hotpotqa_v4a"
    elif lr == 5e-6:
        return "/sfs/weka/scratch/ks8vf/exp/runs/hotpotqa_v4b"
    else:
        return f"/sfs/weka/scratch/ks8vf/exp/runs/hotpotqa_v4_lr{lr:.0e}"


def build_config_blueprint(lr: float) -> chz.Blueprint[train.Config]:
    log_path = get_log_path(lr)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=DATA_FILE,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": MODEL_NAME,
            "dataset_builder": dataset,
            "learning_rate": lr,
            "lr_schedule": "linear",
            "num_epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "save_every": SAVE_EVERY,
            "eval_every": SAVE_EVERY,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, required=True,
                        help="Learning rate (1e-5 for v4a, 5e-6 for v4b)")
    args, remaining = parser.parse_known_args()

    log_path = get_log_path(args.lr)

    print(f"Training config (v4 - LR={args.lr:.0e}):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Data: {DATA_FILE}")
    print(f"  LR: {args.lr:.2e}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Save every: {SAVE_EVERY} steps")
    print(f"  Log path: {log_path}")

    blueprint = build_config_blueprint(args.lr)
    sys.argv = [sys.argv[0]] + remaining
    blueprint.make_from_argv(remaining)
    config = blueprint.make()

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
