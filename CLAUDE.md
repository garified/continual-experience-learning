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

## Eval
- HELMET evals run from `HELMET/` directory using OpenRouter API.
- See `note.md` "Eval Commands" section for reproducible commands.
- Model: `qwen/qwen3-30b-a3b-instruct-2507` via OpenRouter, tokenizer: `Qwen/Qwen3-30B-A3B-Instruct-2507`.
