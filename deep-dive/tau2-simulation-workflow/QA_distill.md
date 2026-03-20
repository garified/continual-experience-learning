# tau2-bench Simulation Workflow: Distilled Guide

## End-to-End Flow

```
CLI (tau2 run --domain retail ...)
  |
  v
TextRunConfig                          # cli.py:663
  |
  v
run_domain(config)                     # runner/batch.py — top-level entry
  |-- load tasks from registry
  |-- create save directory
  v
run_tasks(config, tasks)               # runner/batch.py — batch execution
  |-- ThreadPoolExecutor(max_workers=3)
  |-- for each (task, trial):
  v
run_single_task(config, task, seed)    # runner/batch.py — per-task w/ retry
  |
  v
build_orchestrator(config, task)       # runner/build.py — Layer 2
  |-- build_environment("retail")      → Environment(tools=RetailTools, policy=...)
  |-- build_agent("llm_agent", env)    → LLMAgent(tools, policy, llm, llm_args)
  |-- build_user("user_simulator", task) → UserSimulator(llm, instructions=task.scenario)
  v
Orchestrator(agent, user, env, task)   # orchestrator/orchestrator.py
  |
  v
run_simulation(orchestrator)           # runner/simulation.py — Layer 1
  |-- orchestrator.run()               → SimulationRun
  |-- evaluate_simulation(sim, task)   → RewardInfo
  v
SimulationRun (with reward_info)
```

## The Simulation Loop (orchestrator.run)

```
initialize():
  1. agent.get_init_state()     → system prompt with domain policy
  2. user.get_init_state()      → system prompt with simulation guidelines + task scenario
  3. user.generate_next_message(init_msg)  → first customer utterance
  4. Set routing: from=USER, to=AGENT

while not done:
  step():
    Route message based on (from_role, to_role):

    USER → AGENT:
      agent_msg = agent.generate_next_message(user_msg)  # LLM call
      if tool_call → next: AGENT→ENV
      else         → next: AGENT→USER

    AGENT → ENV:
      results = environment.execute_tool_calls(tool_calls)  # direct function call
      next: ENV→AGENT  (return results)

    AGENT → USER:
      user_msg = user.generate_next_message(agent_msg)  # LLM call
      if "[STOP]" in user_msg → done=True
      if tool_call → next: USER→ENV
      else         → next: USER→AGENT

  check_termination():
    if steps >= 200 → done
    if errors >= 10 → done
    if timeout      → done

finalize():
  → SimulationRun(messages=trajectory, termination_reason, costs)
```

## Key Components

### LLMAgent (`agent/llm_agent.py`)
- **System prompt**: Domain policy document (e.g., retail customer service rules)
- **generate_next_message()**: Appends incoming msg to state, calls `litellm.completion()` with tools, returns AssistantMessage with text or tool_calls
- **is_stop()**: Checks for stop tokens in agent output

### UserSimulator (`user/user_simulator.py`)
- **System prompt**: Global simulation guidelines + task-specific scenario (e.g., "You want to cancel order #W123")
- **generate_next_message()**: Calls LLM with `flip_roles()` (so LLM generates as "assistant" but output becomes UserMessage)
- **Stop signals**: `[STOP]`, `[TRANSFER]`, `[OUT OF SCOPE]` → ends simulation

### Environment (`environment/environment.py`)
- **Tools**: Domain-specific functions (retail has 16: get_order_details, cancel_pending_order, etc.)
- **use_tool()**: Executes tool function, returns ToolMessage with JSON result or error
- **Database**: Mutable state (orders, users, products) that tools read/modify

### LLM Utils (`utils/llm_utils.py`)
- **generate()**: Central LLM call wrapper → `litellm.completion(model, messages, tools, **kwargs)`
- Model string passed directly to LiteLLM (e.g., `openrouter/qwen/qwen3-30b-a3b-instruct-2507`)
- `llm_args` dict passed as `**kwargs` (can include `api_base`, `api_key`, `temperature`)

## Evaluation Pipeline

After simulation completes, `evaluate_simulation()` computes reward as a **product** of components:

```
reward = DB_match × Action_check × Communicate_check × NL_assertions
```

| Component | How it works | Score |
|-----------|-------------|-------|
| **DB Match** | Compare final DB state vs expected state | 0 or 1 |
| **Action Check** | Verify correct tool calls were made with right args | 0-1 (partial) |
| **Communicate** | Verify agent communicated required info to user | 0 or 1 |
| **NL Assertions** | LLM judges natural language conditions against trajectory | 0 or 1 |

Which components apply depends on the task's `reward_basis` field. If any component is 0, total reward is 0.

## Typical Conversation Flow (Retail)

```
[USER→AGENT]  "Hi, I want to cancel my order #W9876543"
[AGENT→ENV]   get_user_details(user_id="...")          → {name, orders, ...}
[ENV→AGENT]   {"user_id": "...", "orders": [...]}
[AGENT→ENV]   get_order_details(order_id="W9876543")   → {status: "pending", items: [...]}
[ENV→AGENT]   {"order_id": "W9876543", "status": "pending", ...}
[AGENT→USER]  "I found your order. It's currently pending. To confirm, you'd like to cancel it?"
[USER→AGENT]  "Yes, please cancel it"
[AGENT→ENV]   cancel_pending_order(order_id="W9876543", reason="customer request")
[ENV→AGENT]   {"success": true, "refund_amount": 49.99}
[AGENT→USER]  "Your order has been cancelled. A refund of $49.99 will be processed."
[USER→AGENT]  "Thanks! [STOP]"
→ done, termination_reason=USER_STOP
```

## Output Structure

```
data/simulations/<timestamp>_<domain>_<agent>_<user>/
  results.json              ← aggregate metrics (avg reward, pass^k)
  tasks/
    task_<id>/
      sim_<uuid>/
        sim_status.json     ← per-sim reward breakdown
        task.log            ← conversation transcript
        llm_debug/          ← raw LLM request/response JSONs
```
