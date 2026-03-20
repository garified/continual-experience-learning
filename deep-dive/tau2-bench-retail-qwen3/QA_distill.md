# Distilled Guide: Running tau2-bench Retail Eval on Qwen3-30B via Tinker API

## Overview

**tau2-bench** evaluates LLM agents on customer service tasks through simulated multi-turn conversations. The agent (model under test) uses tools to resolve customer issues while a user simulator (strong LLM like GPT-4.1) acts as the customer. The **retail** domain tests e-commerce scenarios (order management, returns, exchanges, etc.) with 16 tools.

The system uses **LiteLLM** for all LLM calls, supporting any provider via model prefixes and custom API bases.

---

## Installation

```bash
# Clone
cd /sfs/weka/scratch/ks8vf/exp
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench

# Install (requires Python 3.12+ — uv will handle this automatically)
uv sync

# Configure API keys
cp .env.example .env
# Add to .env: OPENAI_API_KEY=<key>  (needed for user simulator + evaluator)
```

**Python note**: The system Python is 3.11, but `uv` will use Python 3.12 available at `/apps/software/standard/core/jupyterlab/4.4.6-py3.12/bin/python3.12`.

---

## Running the Eval

### Key architecture insight
tau2-bench takes two model specifications:
- `--agent-llm` — The model being evaluated (your Qwen3-30B)
- `--user-llm` — The simulated customer (keep as GPT-4.1 for quality)

LLM args (temperature, api_base, api_key) are passed as JSON dicts via `--agent-llm-args` / `--user-llm-args` and flow directly to `litellm.completion(**kwargs)`.

### Preferred approach: Tinker OAI endpoint + LiteLLM `openai/` prefix

```bash
export TINKER_API_KEY="tml-VN0XMKpmtu12TBT5f2vn5tOvkaBr4sv2Gm5lISSsybZ8qSJNdxar0vYprsmHf3LbBAAAA"
export OPENAI_API_KEY="<your-openai-key>"   # For user-llm (gpt-4.1) + evaluators

cd /sfs/weka/scratch/ks8vf/exp/tau2-bench

uv run tau2 run \
  --domain retail \
  --agent-llm "openai/Qwen/Qwen3-30B-A3B-Instruct-2507" \
  --agent-llm-args '{"api_base": "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", "api_key": "'"$TINKER_API_KEY"'", "temperature": 0.0}' \
  --user-llm gpt-4.1 \
  --num-trials 1 \
  --num-tasks 5 \
  --verbose-logs \
  2>&1 | tee /sfs/weka/scratch/ks8vf/exp/runs/tau2_retail_qwen3_30b.log
```

### Fallback: OpenRouter (if Tinker OAI is unreliable)

```bash
export OPENROUTER_API_KEY="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"

uv run tau2 run \
  --domain retail \
  --agent-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" \
  --agent-llm-args '{"api_key": "'"$OPENROUTER_API_KEY"'", "temperature": 0.0}' \
  --user-llm gpt-4.1 \
  --num-trials 1
```

---

## Model Choice: Base vs Instruct

| Model | Tool Calling | Expected Score | Use Case |
|-------|-------------|---------------|----------|
| `Qwen/Qwen3-30B-A3B` (base) | No training | ~0% | Lower-bound baseline only |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | Yes | Meaningful | Actual evaluation |

**Recommendation**: Use the **Instruct** model. tau2-bench requires structured tool calling — base models will fail to format tool calls correctly and score near zero.

---

## Tinker OAI Endpoint Details

- **Base URL**: `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
- **Auth**: Tinker API key passed as OpenAI API key
- **Endpoints**: `/chat/completions` (uses HuggingFace chat template), `/completions`
- **Model naming**: Documented format is `tinker://UUID:train:0/sampler_weights/NNNNNN` (checkpoint paths). Base model names (`Qwen/Qwen3-30B-A3B`) may or may not work — test first.
- **Limitations**: Beta, low throughput, latency may vary

### If base model names don't work with OAI endpoint:
```python
# Get a sampler checkpoint path for base model
import tinker
sc = tinker.ServiceClient()
sampling_client = sc.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
# Use the resulting tinker:// path in --agent-llm
```

---

## API Keys Required

| Purpose | Key | Notes |
|---------|-----|-------|
| Agent LLM (Qwen3-30B) | `TINKER_API_KEY` | Passed via `--agent-llm-args` |
| User simulator (GPT-4.1) | `OPENAI_API_KEY` | In `.env` or environment |
| NL assertions evaluator | `OPENAI_API_KEY` | Uses `gpt-4.1-2025-04-14` |
| User simulator evaluator | `ANTHROPIC_API_KEY` | Uses `claude-opus-4-5` (optional) |

---

## Results

- Saved to `tau2-bench/data/simulations/`
- View interactively: `uv run tau2 view`
- Metrics computed automatically after run completes
- Re-evaluate trajectories: `uv run tau2 evaluate-trajs`

---

## Step-by-Step Execution Plan

1. `git clone https://github.com/sierra-research/tau2-bench` into `/sfs/weka/scratch/ks8vf/exp/`
2. `cd tau2-bench && uv sync`
3. `cp .env.example .env` and add `OPENAI_API_KEY`
4. Test connectivity with mock domain: `uv run tau2 run --domain mock --num-tasks 1`
5. Test Tinker OAI endpoint accepts model name (quick Python script)
6. Run retail eval with Qwen3-30B-Instruct via Tinker
7. View results with `uv run tau2 view`
