"""
SFT training for paper CPT v6: single-paper knowledge extraction.
15 paraphrase variants per chunk + originals = 352 samples.

Usage:
    python scripts/train_paper_cpt_v6.py
"""

import chz
import asyncio
from pathlib import Path
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig


MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
LEARNING_RATE = 1e-5
BATCH_SIZE = 32
LORA_RANK = 64
NUM_EPOCHS = 4
SAVE_EVERY = 12

DATA_FILE = "/sfs/weka/scratch/ks8vf/exp/data/paper_cpt/v6_augmented.jsonl"
LOG_PATH = "/sfs/weka/scratch/ks8vf/exp/runs/paper_cpt_v6"


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}\nRun paraphrase generation first.")

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
    print(f"Training config (paper CPT v6):")
    print(f"  Model: {config.model_name}")
    print(f"  Data: {DATA_FILE}")
    print(f"  LR: {config.learning_rate:.2e}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  LoRA rank: {config.lora_rank}")
    print(f"  Batch size: {config.dataset_builder.common_config.batch_size}")
    print(f"  Save every: {config.save_every} steps")
    print(f"  Log path: {config.log_path}")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    import sys
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
