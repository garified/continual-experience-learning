"""
SFT training on HotpotQA data v3: Multi-variant passage synthesis.

Experiments:
- v3a: 5 variants per passage
- v3b: 10 variants per passage
- v3c: 15 variants per passage

Usage:
    python scripts/train_hotpotqa_sft_v3.py --variant 5   # v3a
    python scripts/train_hotpotqa_sft_v3.py --variant 10  # v3b
    python scripts/train_hotpotqa_sft_v3.py --variant 15  # v3c
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


# Config (same as v2)
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA_DIR = "/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_v3"

# Conservative hyperparameters (same as v2)
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
LORA_RANK = 64
NUM_EPOCHS = 4
SAVE_EVERY = 12  # Adjust based on total steps


def get_config(num_variants: int) -> tuple[str, str]:
    """Get data file and log path for variant count."""
    data_file = f"{DATA_DIR}/combined_{num_variants}var.jsonl"
    log_path = f"/sfs/weka/scratch/ks8vf/exp/runs/hotpotqa_v3_{num_variants}var"
    return data_file, log_path


def build_config_blueprint(num_variants: int) -> chz.Blueprint[train.Config]:
    data_file, log_path = get_config(num_variants)
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
        file_path=data_file,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": log_path,
            "model_name": MODEL_NAME,
            "dataset_builder": dataset,
            "learning_rate": LEARNING_RATE,
            "lr_schedule": "linear",
            "num_epochs": NUM_EPOCHS,
            "lora_rank": LORA_RANK,
            "save_every": SAVE_EVERY,
            "eval_every": SAVE_EVERY,
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=int, required=True, choices=[5, 10, 15],
                        help="Number of variants per passage (5, 10, or 15)")
    args, remaining = parser.parse_known_args()

    data_file, log_path = get_config(args.variant)

    print(f"Training config (v3 - {args.variant} variants):")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Data: {data_file}")
    print(f"  LR: {LEARNING_RATE:.2e}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  LoRA rank: {LORA_RANK}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Save every: {SAVE_EVERY} steps")
    print(f"  Log path: {log_path}")

    blueprint = build_config_blueprint(args.variant)
    # Pass remaining args to chz
    sys.argv = [sys.argv[0]] + remaining
    blueprint.make_from_argv(remaining)
    config = blueprint.make()

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    main()
