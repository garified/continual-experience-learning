# tau2-bench Train/Test Split: Distilled Guide

## Split Architecture

tau2-bench separates tasks via **external split files** — JSON dicts mapping split names to lists of task IDs. The `Task` model itself carries no split label.

```
data/tau2/domains/<domain>/split_tasks.json   ← defines splits
data/tau2/domains/<domain>/tasks.json         ← all tasks (no split info)
```

## Split Sizes

| Domain   | Train | Test | Base | Other |
|----------|-------|------|------|-------|
| Retail   | 74    | 40   | 114  | —     |
| Airline  | 30    | 20   | 50   | —     |
| Telecom  | 74    | 40   | 114  | small (20), full (2285) |
| Mock     | —     | —    | 10   | —     |
| Banking  | —     | —    | N/A  | No splits yet |

- Train ∩ Test = **0** (fully disjoint in all domains)
- Base = Train ∪ Test (always)

## Loading Flow

```
CLI --task-split-name "train"
  → runner/helpers.py: load_tasks(domain, split_name)
    → registry.get_tasks_loader(domain)(task_split_name="train")
      → domains/retail/environment.py: get_tasks("train")
        → loads ALL tasks from tasks.json
        → loads split_tasks.json
        → filters: [t for t in tasks if t.id in splits["train"]]
        → returns 74 tasks
```

## CLI Usage

```bash
# Default: base split (all 114 tasks)
tau2 run --domain retail

# Train only (74 tasks)
tau2 run --domain retail --task-split-name train

# Test only (40 tasks, held-out)
tau2 run --domain retail --task-split-name test

# Specific IDs (overrides split)
tau2 run --domain retail --task-ids 5 9 12

# First N from a split
tau2 run --domain retail --task-split-name test --num-tasks 10
```

## Key Details

1. **Default is `base`** — runs on ALL tasks (train + test). The `base` list concatenates train IDs first, then test IDs.

2. **`--num-tasks N` takes from the front** — with base split, small N values always select train tasks (IDs 0, 1, 2, ...).

3. **`--task-ids` applied after split** — `--task-split-name test --task-ids 5 9` filters within test.

4. **Split file naming convention**: `split_<task_file_stem>.json` adjacent to the task data file.

## For Fine-Tuning Workflow

```bash
# Step 1: Generate SFT trajectories on TRAIN tasks
tau2 run --domain retail --task-split-name train \
  --agent-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" \
  --user-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507"

# Step 2: Evaluate fine-tuned model on TEST tasks (no leakage)
tau2 run --domain retail --task-split-name test \
  --agent-llm "<fine-tuned-model>" \
  --user-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
```
