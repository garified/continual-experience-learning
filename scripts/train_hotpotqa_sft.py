"""
SFT training on HotpotQA data for knowledge extraction.

Experiment v1: Type1 (passage absorption) + Type2 (multi-hop QA, no gold leakage)
"""

import chz
import sys
import asyncio
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


# Config
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
DATA_FILE = "/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_20/combined_train.jsonl"
LOG_PATH = "/sfs/weka/scratch/ks8vf/exp/runs/hotpotqa_v2"

# v2: Conservative hyperparameters based on previous successful Tinker SFT
LEARNING_RATE = 1e-5  # Much lower than auto (5e-4)
BATCH_SIZE = 32  # Smaller batches
LORA_RANK = 64
NUM_EPOCHS = 4
SAVE_EVERY = 12  # ~12 checkpoints for 144 steps (36 batches × 4 epochs)


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL_NAME,
        renderer_name=renderer_name,
        max_length=4096,  # Passages are relatively short
        batch_size=BATCH_SIZE,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )

    dataset = FromConversationFileBuilder(
        common_config=common_config,
        file_path=DATA_FILE,
    )

    return chz.Blueprint(train.Config).apply(
        {
            "log_path": LOG_PATH,
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


def main(config: train.Config):
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    print(f"Training config:")
    print(f"  Model: {config.model_name}")
    print(f"  LR: {config.learning_rate:.2e}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Batch size: {config.dataset_builder.common_config.batch_size}")
    print(f"  Save every: {config.save_every} steps")
    print(f"  Log path: {config.log_path}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
