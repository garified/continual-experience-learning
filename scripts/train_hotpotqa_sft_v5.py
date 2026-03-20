"""
SFT training for v5: per-slice knowledge extraction (5 x 20 questions).

Usage:
    python scripts/train_hotpotqa_sft_v5.py --slice 1
    python scripts/train_hotpotqa_sft_v5.py --slice 2
"""

import argparse
import chz
import sys
import asyncio
from pathlib import Path
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


# Config (same as v2 except 2 epochs)
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
LORA_RANK = 64
NUM_EPOCHS = 2
SAVE_EVERY = 12


def get_data_file(slice_num: int) -> str:
    """Get data file path for a slice. Slice 1 reuses v2 data."""
    if slice_num == 1:
        return "/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_20/combined_train.jsonl"
    return f"/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_v5_s{slice_num}/combined_train.jsonl"


def get_log_path(slice_num: int) -> str:
    return f"/sfs/weka/scratch/ks8vf/exp/runs/hotpotqa_v5_s{slice_num}"


def build_config_blueprint(slice_num: int) -> chz.Blueprint[train.Config]:
    data_file = get_data_file(slice_num)
    log_path = get_log_path(slice_num)

    if not Path(data_file).exists():
        raise FileNotFoundError(f"Data file not found: {data_file}\nRun: python scripts/prep_hotpotqa_data_v5.py --slice {slice_num}")

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


def main(config: train.Config, slice_num: int):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    print(f"Training config (v5 slice {slice_num}):")
    print(f"  Model: {config.model_name}")
    print(f"  LR: {config.learning_rate:.2e}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Batch size: {config.dataset_builder.common_config.batch_size}")
    print(f"  Save every: {config.save_every} steps")
    print(f"  Log path: {config.log_path}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train v5 slice model")
    parser.add_argument("--slice", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Slice number (1-5)")
    args, remaining = parser.parse_known_args()

    blueprint = build_config_blueprint(args.slice)
    blueprint.make_from_argv(remaining)
    main(blueprint.make(), args.slice)
