# Research Plan: Fine-Tune LLM to Replace Long-Context Inference

## Goal
Fine-tune an LLM on long-context content so it can answer questions **without** the context at inference time, matching the performance of in-context long-context eval.

## Eval: HELMET
Benchmark with 7 task types: Recall, RAG, Re-ranking, Citation, LongQA, Summarization, ICL.
Lengths: 8K–128K. Paper: https://arxiv.org/abs/2410.02694

## Plan
1. **Baseline**: Run HELMET with full context in-context. Record scores per task/length.
2. **Fine-tune**: Train on the context content (continued pre-training, SFT on QA pairs, or both). Use LoRA for efficiency.
3. **Eval**: Run HELMET again without context. Compare against baseline.
4. **Analyze**: Which tasks internalize well vs. still need retrieval? How does context length affect the gap?

- 2026-02-11: Follow Physics of LM 3.1 recipe for knowledge extraction — mixed pre-training + QA is effective for OOD knowledge extraction. Multiple textual variants mentioning the same fact (e.g., bios, paraphrases) help more than a single piece of training data about that fact.
- 2026-02-11: Data plan for HotpotQA SFT on Qwen3-30B-A3B-Instruct-2507 via Tinker. Train/dev passage overlap is only 0.6%, so train-split QA is useless for dev knowledge. Instead: (1) Type 1 — absorb all 20 `ctxs` passages per sample as chat pairs (knowledge absorption), (2) Type 2 — teacher model scans 20 passages to discover naturally linked pairs (NOT using gold `positive_ctxs` — that's leakage), then generates many multi-hop QA per linked pair (e.g., ~8 pairs × ~10 questions = ~80 QA per sample), (3) Eval — HELMET dev questions without context.
- 2026-02-11: Type 1 data currently does not track which passages belong to which eval prompt. Future improvement: tag passages by source prompt and add direct questions on entire prompt context (similar to prompt distillation approach).

## Log
- 2026-02-11: Initialized git repo in `exp/`. Cloned HELMET and tinker-cookbook are `.gitignore`d; only our own files (note.md, scripts, configs, results) are tracked.
- 2026-02-11: Ran HELMET RAG baseline on `kilt_hotpotqa` with Qwen3-30B-A3B-Instruct-2507 via OpenRouter (`--use_vllm_serving --endpoint_url https://openrouter.ai/api/v1/`). Config: `configs/rag.yaml`, 128K context, 100 questions × 3 depth levels = 300 samples, 2-shot, `max_workers=20`. Required fixes: added `--tokenizer_name_or_path` for OpenRouter, fixed `stop_newline`→`stop_new_line` bug, added null guard for empty API responses. Results: **EM 44.41, F1 54.30, substring_EM 51.19, rougeL 53.96**. Output: `HELMET/output/qwen3-30b-a3b-instruct-2507/kilt_hotpotqa_eval_hotpotqa-dev-multikilt_1000_k1000_dep3_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatTrue_42.json`.
- 2026-02-11: Added `--no_context` flag to HELMET for zero-shot (no documents, no demos) eval. Changes in `arguments.py` (new flag), `data.py:load_qa()` (skip context/demos, use clean prompt without "Use the given documents"), and `data.py:load_data()` (pass flag through). Running zero-shot eval on kilt_hotpotqa with same Qwen3-30B-A3B model to measure parametric knowledge baseline.
- 2026-02-11: Zero-shot (no context) eval completed on kilt_hotpotqa with Qwen3-30B-A3B-Instruct-2507. Results: **EM 25.00, F1 37.99, substring_EM 35.33, rougeL 38.43** (avg input 54 tokens, 14s total). Compared to 128K-context baseline (EM 44.41, F1 54.30), the context adds ~19 EM / ~16 F1 points. The model's parametric knowledge already covers 25% of answers. Output: `HELMET/output/qwen3-30b-a3b-instruct-2507/kilt_hotpotqa_no_context_hotpotqa-dev-multikilt_1000_k1000_dep3_in131072_size100_shots2_sampFalsemax20min0t0.0p1.0_chatTrue_42.json`.
- 2026-02-11: Data prep script `scripts/prep_hotpotqa_data.py` updated to use OpenRouter with correct model name `qwen/qwen3-30b-a3b-instruct-2507` and parallel inference (10 concurrent calls).
- 2026-02-11: Fixed data leakage in Type 2 QA generation — now excludes any links involving gold `positive_ctxs` passages. Regenerated data: Type1=400, Type2=740 (down from 2510), Combined=1140, Ratio=1.85× (down from 6.3×). Old data backed up to `data/hotpotqa_20_old_with_leakage/`.
- 2026-02-11: v1 SFT training completed (32 steps, 4 epochs, LR=5e-4, LoRA rank=32, batch=128). Train NLL: 2.76→1.01. Eval on 300 HotpotQA questions (zero-shot): all checkpoints perform **worse** than base model (EM 0-2% vs 25%, F1 18-21% vs 38%). Model appears to repeat training data patterns instead of generalizing. Hypothesis: Type1+Type2 data lacks diversity, or QA format doesn't transfer to eval format.
- 2026-02-11: v1 hyperparameters likely too aggressive. Previous successful Tinker SFT used LR=1e-5, batch=32, LoRA=64. v1 used LR=5e-4 (50× higher), batch=128 (4× larger). Next run (v2): LR=1e-5, batch=32, save_every=12 for 12 checkpoints.
- 2026-02-11: v2 SFT completed (140 steps, 4 epochs, NLL 2.6→1.74). Initial eval on 300 samples showed EM 7-9% — but this was misleading: trained on 20 questions, evaluated on 100.
- 2026-02-11: **Corrected eval**: On 60 samples (20 trained questions × 3 depths), v2 achieves **EM 45%, F1 54%** — nearly matching RAG baseline (EM 44.41, F1 54.30) without any context! Knowledge extraction worked. Earlier issues: (1) prompt mismatch with HELMET, (2) "Answer:" prefix not stripped, (3) eval scope too broad.
- 2026-02-11: Split analysis on trained vs untrained questions: v2 final on first 60 (trained) = EM 45, F1 53; rest 240 (untrained) = EM 17.5, F1 30; all 300 = EM 22, F1 34. Baseline zero-shot: first 60 = EM 23, F1 36; rest 240 = EM 25, F1 38. **Conclusion**: SFT boosts trained questions by +22 EM pts but causes mild forgetting on untrained (-8 EM pts).
- 2026-02-11: v3 data synthesis completed. Following Physics of LM 3.1, generated 15 paraphrase variants per passage using explicit styles (formal academic, casual blog, bullet points, Q&A, narrative, technical doc, news article, Wikipedia, textbook, dialogue, executive summary, detailed elaboration, first person, timeline, compare/contrast). Data: Type1 all=6000, Type2=720. Combined datasets: 5var=2720, 10var=4720, 15var=6720 samples.

- 2026-02-11: v4 eval completed on all checkpoints (`results/v4_all_checkpoints.json`). v4a best: EM 26.33, F1 37.08 (step 408); v4b best: EM 26.67, F1 36.75 (step 840). Both plateau around EM ~25, F1 ~34 after ~300 steps. Tinker base model (zero-shot, temp=0) eval: **EM 16.33, F1 31.31** (`results/base_model_eval.json`). Note: Tinker base model score (EM 16.33) differs from HELMET OpenRouter baseline (EM 25.00) — likely due to prompt format differences (Tinker uses chat template vs HELMET's 2-shot RAG prompt without context). v4 training adds ~+10 EM over the Tinker baseline.

## Conceptual Corrections

Project-specific corrections to prevent repeated misunderstandings.

- 2026-02-11: Rest 240's EM drop (25.4→17.5) is NOT "failure to generalize to new questions about same passages." HELMET HotpotQA gives each question its own unique 1000-passage set. Rest 240 = 80 different questions with entirely different passages never seen in training. The drop reflects degraded general QA ability from overfitting, not knowledge extraction failure.

## Eval Commands

All evals run from `HELMET/` directory. Requires `export OPENROUTER_API_KEY=...`.

**With context (128K RAG baseline):**
```bash
python eval.py --config configs/rag.yaml \
  --datasets kilt_hotpotqa \
  --test_files data/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl \
  --demo_files data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl \
  --model_name_or_path qwen/qwen3-30b-a3b-instruct-2507 \
  --use_vllm_serving \
  --endpoint_url "https://openrouter.ai/api/v1/" \
  --api_key "$OPENROUTER_API_KEY" \
  --tokenizer_name_or_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --use_chat_template True --overwrite
```

**Zero-shot (no context, parametric knowledge only):**
```bash
python eval.py --config configs/rag.yaml \
  --datasets kilt_hotpotqa \
  --test_files data/kilt/hotpotqa-dev-multikilt_1000_k1000_dep3.jsonl \
  --demo_files data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl \
  --model_name_or_path qwen/qwen3-30b-a3b-instruct-2507 \
  --use_vllm_serving \
  --endpoint_url "https://openrouter.ai/api/v1/" \
  --api_key "$OPENROUTER_API_KEY" \
  --tokenizer_name_or_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --use_chat_template True --no_context --tag no_context --overwrite
```

Note: `configs/rag.yaml` sets `max_test_samples=100`, `shots=2`, `input_max_length=131072`. The test file has `dep3` (3 depth levels per question), so 100 questions × 3 = 300 samples.

## Experiments

Each experiment variant is tracked as a "time slice" with consistent data/model/results.

| Exp | Strategy | Data Script | Data File | Training Checkpoint | Eval Results | Notes |
|-----|----------|-------------|-----------|---------------------|--------------|-------|
| v0 | baseline (128K context) | — | — | — | EM 44.41, F1 54.30 | Upper bound |
| v0 | zero-shot (no context) | — | — | — | EM 25.00, F1 37.99 | Lower bound |
| v1 | Type1+Type2 (no gold leakage) | `scripts/prep_hotpotqa_data.py` | `data/hotpotqa_20/combined_train.jsonl` (1140) | `runs/hotpotqa_v1` step8/16/24/final | EM 0-2, F1 18-21 | **Failed** — LR=5e-4, batch=128 too aggressive |
| v2 | Type1+Type2 (conservative HP) | `scripts/prep_hotpotqa_data.py` | `data/hotpotqa_20/combined_train.jsonl` (1140) | `runs/hotpotqa_v2` 12 ckpts | **EM 45, F1 54** (60 samples) | LR=1e-5, batch=32, LoRA=64 — **matches RAG baseline!** |
| v3a | Type1 (5 variants) + Type2 | `scripts/prep_hotpotqa_data_v3.py` | `data/hotpotqa_v3/combined_5var.jsonl` (2720) | `runs/hotpotqa_v3_5var` 29 ckpts | TBD | 339 steps, NLL 1.38 |
| v3b | Type1 (10 variants) + Type2 | `scripts/prep_hotpotqa_data_v3.py` | `data/hotpotqa_v3/combined_10var.jsonl` (4720) | `runs/hotpotqa_v3_10var` 49 ckpts | TBD | 587 steps, NLL 1.21 |
| v3c | Type1 (15 variants) + Type2 | `scripts/prep_hotpotqa_data_v3.py` | `data/hotpotqa_v3/combined_15var.jsonl` (6720) | `runs/hotpotqa_v3_15var` 70 ckpts | TBD | 839 steps, NLL 1.17 |
| v4a | Type1+Type2 (100 questions) | `scripts/prep_hotpotqa_data_v4.py` | `data/hotpotqa_100/combined_train.jsonl` | `runs/hotpotqa_v4a` | Best: **EM 26.33, F1 37.08** (step 408); Final: EM 25.00, F1 33.57 | LR=1e-5, same as v2 but all 100 questions |
| v4b | Type1+Type2 (100 questions) | `scripts/prep_hotpotqa_data_v4.py` | `data/hotpotqa_100/combined_train.jsonl` | `runs/hotpotqa_v4b` | Best: **EM 26.67, F1 36.75** (step 840); Final: EM 24.33, F1 35.29 | LR=5e-6, half of v4a |

## v2 Detailed Results (Trained vs Untrained Split)

| Model | First 60 (trained) | Rest 240 (untrained) | All 300 |
|-------|-------------------|---------------------|---------|
| Baseline (zero-shot) | EM 23.33, F1 36.49 | EM 25.42, F1 38.37 | EM 25.00, F1 37.99 |
| v2 step 12 | EM 35.00, F1 47.56 | EM 12.50, F1 27.04 | EM 18.00, F1 32.06 |
| v2 step 72 | EM 45.00, F1 54.03 | EM 15.00, F1 27.61 | EM 21.00, F1 32.91 |
| v2 final | EM 45.00, F1 53.43 | EM 17.50, F1 30.09 | EM 22.00, F1 33.83 |
| RAG baseline (128K) | EM 46.67, F1 54.88 | EM 43.83, F1 54.16 | EM 44.41, F1 54.30 |

## Core Question
Where is the boundary between knowledge that can be baked into weights vs. knowledge that must be retrieved at inference time?
