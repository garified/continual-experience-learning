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

## Core Question
Where is the boundary between knowledge that can be baked into weights vs. knowledge that must be retrieved at inference time?
