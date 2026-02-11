# Project: Knowledge Extraction via Fine-Tuning

## Git
- The primary repo is `exp/` (this directory). All commits should be made here unless explicitly asked otherwise.
- `HELMET/` and `tinker-cookbook/` are gitignored subprojects with their own repos.
- When committing, stage `note.md`, `scripts/`, `data/`, `runs/` as appropriate. Avoid committing large files (model checkpoints, `.tar.gz`).

## Structure
- `note.md` — Research log, plans, experiment tracking. Use `/log` skill to append entries.
- `scripts/` — Data prep and eval scripts.
- `data/` — Generated training data (Type1 passages, Type2 QA).
- `runs/` — Tinker training runs and eval results.
- `HELMET/` — Cloned benchmark repo (gitignored). Our modifications tracked in its own git.

## Experiment Versioning
When a new idea, strategy, or hypothesis leads to new data synthesis or training:
1. **Create versioned files** — never overwrite previous versions:
   - Data script: `scripts/prep_hotpotqa_data_v{N}.py`
   - Training script: `scripts/train_hotpotqa_sft_v{N}.py`
   - Eval script: `scripts/eval_checkpoint_hotpotqa_v{N}.py`
   - Data directory: `data/hotpotqa_v{N}/`
   - Training run: `runs/hotpotqa_v{N}/`
2. **Update `note.md`** — add a row to the `## Experiments` table with the new version's Strategy, Data Script, Data File, Training Checkpoint, Eval Results, and Notes.
3. Previous versions are kept intact for reproducibility and comparison.

## Eval
- HELMET evals run from `HELMET/` directory using OpenRouter API.
- See `note.md` "Eval Commands" section for reproducible commands.
- Model: `qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter, tokenizer: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
