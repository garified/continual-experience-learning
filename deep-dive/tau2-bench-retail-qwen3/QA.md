# Deep Dive: Running tau2-bench Retail on Qwen3-30B via Tinker API

## Q1: What is tau2-bench and how is it structured?

**tau2-bench** (also called tau-bench or tau3-bench) is a simulation framework for evaluating customer service agents. It simulates multi-turn conversations between an **agent** (the model being evaluated) and a **user** (simulated by another LLM), where the agent must use tools to resolve customer service tasks.

### Architecture:
```
CLI (cli.py)
  -> RunConfig (TextRunConfig / VoiceRunConfig)
    -> runner/batch.py: run_domain() -> run_tasks() -> run_single_task()
      -> runner/build.py: build_orchestrator() -> build_agent() + build_user()
        -> runner/simulation.py: run_simulation()
```

### Key directories:
- `src/tau2/agent/` — Agent implementations (LLMAgent for text, DiscreteTimeAudioNativeAgent for voice)
- `src/tau2/domains/` — Domain-specific environments (airline, retail, telecom, banking_knowledge, mock)
- `src/tau2/runner/` — Execution layers (batch, build, simulation, checkpoint)
- `src/tau2/utils/llm_utils.py` — LiteLLM wrapper for all LLM calls
- `src/tau2/config.py` — Default constants
- `src/tau2/data_model/simulation.py` — RunConfig classes

### LLM Integration:
All LLM calls go through `utils/llm_utils.py:generate()` which calls `litellm.completion()`. The model string and kwargs are passed directly. LiteLLM handles routing to the appropriate provider based on model prefix (e.g., `openai/` for OpenAI-compatible endpoints).

---

## Q2: What is the retail domain and what tools does it provide?

The **retail** domain simulates an e-commerce customer service environment. Files:
- `src/tau2/domains/retail/data_model.py` — Data models (Product, User, Order, Variant, payment types)
- `src/tau2/domains/retail/environment.py` — Environment setup with RetailDB, policy loading, task management
- `src/tau2/domains/retail/tools.py` — 16 tools available to the agent
- `src/tau2/domains/retail/utils.py` — Helpers

### Available tools:
**Read-only:**
- `find_user_id_by_email` — Find user by email
- `find_user_id_by_name_zip` — Find user by name + zip
- `get_order_details` — Order status/details
- `get_product_details` — Product inventory
- `get_item_details` — Item inventory
- `get_user_details` — User details including orders
- `list_all_product_types` — List all products

**Write/Modification:**
- `cancel_pending_order` — Cancel + refund
- `modify_pending_order_address` — Update shipping address
- `modify_pending_order_items` — Change items with price adjustment
- `modify_pending_order_payment` — Switch payment method
- `modify_user_address` — Update default address
- `exchange_delivered_order_items` — Item exchange
- `return_delivered_order_items` — Process returns

**Generic:**
- `calculate` — Math expressions
- `transfer_to_human_agents` — Escalation

All write operations require explicit user confirmation.

### Data models:
- **Products** have Variants with item_id, options (color, size), availability, pricing
- **Users** have name, address, email, payment methods (CreditCard, Paypal, GiftCard), order history
- **Orders** track status (processed, pending, delivered, cancelled, exchange/return requested), items, fulfillment, payments

---

## Q3: How does tau2-bench invoke LLMs and how does model configuration flow?

### Configuration flow:
1. **CLI** (`cli.py`) parses `--agent-llm`, `--user-llm`, `--agent-llm-args`, `--user-llm-args`
2. These populate a `TextRunConfig` object:
   ```python
   llm_agent: str = "gpt-4.1-2025-04-14"
   llm_user: str = "gpt-4.1-2025-04-14"
   llm_args_agent: dict = {"temperature": 0.0}
   llm_args_user: dict = {"temperature": 0.0}
   ```
3. `runner/build.py:build_agent()` passes `llm=config.llm_agent, llm_args=config.llm_args_agent` to the agent factory
4. The agent's `generate_next_message()` calls `utils/llm_utils.py:generate(model=self.llm, **self.llm_args, ...)`
5. `generate()` calls `litellm.completion(model=model, messages=..., tools=..., **kwargs)`

### Key insight for custom providers:
The `llm_args` dict is passed as `**kwargs` to `litellm.completion()`. LiteLLM supports:
- `api_base` — Custom endpoint URL
- `api_key` — API key for the endpoint

So you CAN pass custom provider settings through `--agent-llm-args`:
```json
{"temperature": 0.0, "api_base": "https://...", "api_key": "..."}
```

### LiteLLM model naming:
For OpenAI-compatible endpoints, prefix the model with `openai/`:
```
openai/model-name
```
LiteLLM will route to the `api_base` URL using OpenAI client format.

---

## Q4: How does the Tinker API work for inference, and how can it integrate with tau2-bench?

### Tinker OpenAI-Compatible API (Beta):
- **Base URL**: `https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1`
- **Auth**: Pass `TINKER_API_KEY` as the OpenAI API key
- **Endpoints**: `/completions` and `/chat/completions`
- **Model naming**: Uses Tinker sampler weight paths: `tinker://UUID:train:0/sampler_weights/NNNNNN`
- **Chat template**: `/chat/completions` uses the model's default HuggingFace chat template

### The base model problem:
The Tinker OAI endpoint documentation only shows sampler weight paths (fine-tuned checkpoints). However, the Tinker **SDK** supports direct base model inference:
```python
service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B")
```

For the OAI endpoint, you might be able to pass `Qwen/Qwen3-30B-A3B` directly as the model name (undocumented but plausible since the OAI endpoint is a wrapper). Alternatively, you can:
1. Create a training client with 0 training steps
2. Save initial weights to get a sampler path
3. Use that sampler path with the OAI endpoint

### Integration with tau2-bench via LiteLLM:
```bash
tau2 run --domain retail \
  --agent-llm "openai/Qwen/Qwen3-30B-A3B" \
  --agent-llm-args '{"api_base": "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", "api_key": "tml-..."}' \
  --user-llm gpt-4.1 \
  --num-trials 1
```

Or via environment variables:
```bash
export OPENAI_API_BASE=https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1
export OPENAI_API_KEY=tml-...
tau2 run --domain retail --agent-llm "openai/Qwen/Qwen3-30B-A3B" ...
```

**IMPORTANT**: The user-llm (user simulator) should remain a strong model (e.g., gpt-4.1) since it needs to simulate realistic user behavior. Only the agent-llm should be the model under test.

---

## Q5: What are the installation requirements and how to install tau2-bench?

### Requirements:
- **Python 3.12+** (hard requirement in pyproject.toml: `>=3.12,<3.14`)
- **uv** package manager (recommended)
- Core dependencies: Rich, FastAPI, Uvicorn, Pandas, LiteLLM, tenacity, deepdiff, PyYAML

### Installation:
```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
uv sync  # Basic text-mode (sufficient for retail)
```

Optional extras:
- `uv sync --extra knowledge` — For banking_knowledge domain (needs rank-bm25, openai)
- `uv sync --extra voice` — For voice/audio evaluation
- `uv sync --extra gym` — For RL training interface

### Environment setup:
```bash
cp .env.example .env
# Edit .env to add API keys
```

### On your system:
- Python 3.11.6 is the conda default, but **Python 3.12.12** is available at `/apps/software/standard/core/jupyterlab/4.4.6-py3.12/bin/python3.12`
- `uv` is installed at `/home/ks8vf/.local/bin/uv` (v0.7.7)
- `uv sync` will handle creating a virtual environment with the correct Python version

---

## Q6: What CLI options are relevant for running a retail evaluation?

### Core run command:
```bash
tau2 run --domain retail --agent-llm MODEL --user-llm MODEL [options]
```

### Key arguments:
| Argument | Default | Description |
|----------|---------|-------------|
| `--domain` | required | Domain to evaluate (retail, airline, telecom, etc.) |
| `--agent-llm` | `gpt-4.1` | Model for the agent under test |
| `--user-llm` | `gpt-4.1` | Model for user simulator |
| `--agent-llm-args` | `{"temperature": 0.0}` | JSON kwargs passed to agent LLM |
| `--user-llm-args` | `{"temperature": 0.0}` | JSON kwargs passed to user LLM |
| `--num-trials` | 1 | Repetitions per task |
| `--num-tasks` | all | Number of tasks to run (omit for all) |
| `--max-steps` | 200 | Max simulation steps per task |
| `--seed` | 300 | Random seed |
| `--verbose-logs` | false | Detailed logging |
| `--auto-resume` | false | Resume incomplete runs |

### Results:
- Saved to `data/simulations/`
- View with `tau2 view`
- Metrics computed automatically after run completes

### Task splits:
The retail domain has task splits (e.g., "base"). You can specify subsets with task IDs or splits.

---

## Q7: What are the potential issues and gotchas?

### 1. Python version mismatch
tau2-bench requires Python 3.12+. Your conda env uses 3.11. Solution: use `uv` which will find/install the right Python.

### 2. Tinker OAI API model naming
The documented model format is `tinker://UUID:train:0/sampler_weights/NNNNNN`. It's unclear if base model names like `Qwen/Qwen3-30B-A3B` work directly with the OAI endpoint. You may need to:
- Test with the base model name first
- If that fails, create a zero-step training run and save initial weights

### 3. User simulator needs a strong model
The user-llm simulates realistic customers. Using a weak/base model for user simulation will degrade evaluation quality. Keep `--user-llm gpt-4.1` (requires `OPENAI_API_KEY`).

### 4. Tinker OAI API is beta
"Latency and throughput may vary by model and may change without notice during the beta." Not intended for high-throughput use.

### 5. tau2-bench's default evaluator
The `nl_assertions_llm` defaults to `gpt-4.1-2025-04-14` and `user_simulator_evaluator` to `claude-opus-4-5`. These evaluation models also need API keys.

### 6. LiteLLM model prefix
When using `--agent-llm`, you need the `openai/` prefix for LiteLLM to route to the custom endpoint: `openai/Qwen/Qwen3-30B-A3B`.

### 7. Tool calling support
The Qwen3-30B-A3B **base** model may not support tool/function calling well since it hasn't been instruction-tuned. tau2-bench requires the agent to make tool calls. Consider using `Qwen/Qwen3-30B-A3B-Instruct-2507` instead for functional tool calling.

### 8. Cost
The user-llm (gpt-4.1) and evaluator models will incur OpenAI/Anthropic API costs. The agent-llm via Tinker uses your Tinker credits.

---

## Q8: What is the recommended step-by-step procedure to get this running?

### Step 1: Clone and install tau2-bench
```bash
cd /sfs/weka/scratch/ks8vf/exp
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
uv sync
```

### Step 2: Configure environment
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY=<your-openai-key>  # For user simulator + evaluator
```

### Step 3: Test basic connectivity
```bash
# Quick test with mock domain first
uv run tau2 run --domain mock --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 1
```

### Step 4: Test Tinker OAI API connectivity
```bash
# Test if Tinker OAI endpoint accepts base model names
export TINKER_API_KEY="tml-VN0XMKpmtu12TBT5f2vn5tOvkaBr4sv2Gm5lISSsybZ8qSJNdxar0vYprsmHf3LbBAAAA"

python -c "
from openai import OpenAI
client = OpenAI(
    base_url='https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1',
    api_key='$TINKER_API_KEY'
)
# Try base model name
resp = client.chat.completions.create(
    model='Qwen/Qwen3-30B-A3B',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=50
)
print(resp.choices[0].message.content)
"
```

If that fails, get a sampler checkpoint path:
```python
import tinker
sc = tinker.ServiceClient()
sampling_client = sc.create_sampling_client(base_model="Qwen/Qwen3-30B-A3B")
# Note the model_path from sampling_client for use with OAI endpoint
```

### Step 5: Run retail eval with Qwen3-30B via Tinker
```bash
cd /sfs/weka/scratch/ks8vf/exp/tau2-bench

export TINKER_API_KEY="tml-VN0XMKpmtu12TBT5f2vn5tOvkaBr4sv2Gm5lISSsybZ8qSJNdxar0vYprsmHf3LbBAAAA"
export OPENAI_API_KEY="<your-openai-key>"  # For user simulator

uv run tau2 run \
  --domain retail \
  --agent-llm "openai/Qwen/Qwen3-30B-A3B" \
  --agent-llm-args '{"api_base": "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1", "api_key": "'"$TINKER_API_KEY"'"}' \
  --user-llm gpt-4.1 \
  --num-trials 1 \
  --num-tasks 5 \
  --verbose-logs \
  2>&1 | tee /sfs/weka/scratch/ks8vf/exp/runs/tau2_retail_qwen3_30b.log
```

### Step 6: View results
```bash
uv run tau2 view
```

---

## Q9: Should I use Qwen3-30B-A3B (base) or Qwen3-30B-A3B-Instruct-2507?

### The critical consideration:
tau2-bench requires agents to perform **tool calling** — the agent must decide which tools to invoke, format arguments correctly, and interpret results. This is a structured interaction pattern that base models are not trained for.

### Base model (`Qwen/Qwen3-30B-A3B`):
- No instruction following or tool calling training
- Will likely produce incoherent tool calls or refuse to use tools
- Scores will likely be very low (near 0)
- Useful only as a lower-bound baseline

### Instruct model (`Qwen/Qwen3-30B-A3B-Instruct-2507`):
- Trained for instruction following and tool/function calling
- Will produce properly formatted tool calls
- Meaningful evaluation scores
- This is what you'd typically evaluate

### Recommendation:
If you want a meaningful eval, use `Qwen/Qwen3-30B-A3B-Instruct-2507`. If you want a baseline to show improvement from fine-tuning, the base model gives you a floor score.

Note: The Tinker model lineup lists both `Qwen/Qwen3-30B-A3B` (base) and `Qwen/Qwen3-30B-A3B-Instruct-2507` (instruct).

---

## Q10: What alternative approach exists if the Tinker OAI endpoint doesn't work well with tau2-bench?

### Alternative: Use OpenRouter instead
Your project already uses OpenRouter for other evals. OpenRouter supports Qwen3-30B models:

```bash
export OPENROUTER_API_KEY="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"

uv run tau2 run \
  --domain retail \
  --agent-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" \
  --agent-llm-args '{"api_key": "'"$OPENROUTER_API_KEY"'"}' \
  --user-llm gpt-4.1 \
  --num-trials 1
```

LiteLLM has built-in OpenRouter support with the `openrouter/` prefix.

### Alternative: Direct Tinker SDK custom agent
If neither OAI endpoint nor OpenRouter works, you could write a custom tau2 agent that uses the Tinker SDK directly. This is more work but gives full control:
1. Subclass `HalfDuplexAgent`
2. Override `generate_next_message()` to call Tinker's `sampling_client.sample()`
3. Register the agent in tau2's registry

This is the nuclear option — try the LiteLLM integration first.
