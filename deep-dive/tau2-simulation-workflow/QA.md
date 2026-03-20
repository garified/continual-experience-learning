# Deep Dive: tau2-bench Simulation Workflow

## Q1: What happens when you run `tau2 run --domain retail`? What's the entry point?

The entry point is `cli.py:main()`. It parses CLI args and calls `run_command()`, which:

1. Builds a `TextRunConfig` (or `VoiceRunConfig`) from CLI args:
   ```python
   # cli.py:663
   config = TextRunConfig(
       domain=args.domain,
       llm_agent=args.agent_llm,
       llm_args_agent=args.agent_llm_args,
       llm_user=args.user_llm,
       llm_args_user=args.user_llm_args,
       num_trials=args.num_trials,
       ...
   )
   ```

2. Calls `run_domain(config)` — the top-level batch runner in `runner/batch.py`.

`run_domain()` does:
- Validates config and displays it
- Loads tasks via `get_tasks()` which queries the registry
- Optionally filters tasks by agent-specific task filter
- Sets up save directory: `data/simulations/<timestamp>_<domain>_<agent>_<user>/`
- Delegates to `run_tasks(config, tasks, ...)`

---

## Q2: How does `run_tasks()` batch-execute tasks?

`run_tasks()` in `runner/batch.py` handles concurrent execution:

1. **Seed management**: Generates independent random seeds for each trial
2. **Checkpoint**: If `auto_resume=True` and a previous run exists, loads checkpoint to skip completed tasks
3. **Thread pool**: Uses `ThreadPoolExecutor(max_workers=config.max_concurrency)` (default 3)
4. **Per task-trial**: Submits `run_single_task()` as a future
5. **Progress monitoring**: Periodically prints status showing running tasks, completion rate, average reward
6. **Results collection**: Gathers all `SimulationRun` results, computes aggregate metrics, saves `results.json`

```python
# batch.py (simplified)
with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
    for task in tasks:
        for trial in range(config.num_trials):
            future = executor.submit(run_single_task, config, task, seed=seeds[trial])
            futures.append(future)
```

---

## Q3: What does `run_single_task()` do for one task?

`run_single_task()` in `runner/batch.py` is Layer 3 — per-task execution with retry:

1. **Logging setup**: Creates `_TaskLogContext` that configures per-task file logging and LLM debug log directory
2. **Build orchestrator** (Layer 2): `build_orchestrator(config, task, seed=seed)`
3. **Run simulation** (Layer 1): `run_simulation(orchestrator, evaluation_type=...)`
4. **Retry on failure**: Uses `run_with_retry()` wrapper — retries up to `max_retries` times on LLM errors
5. **Hallucination retries**: If enabled, checks for hallucinated tool calls and retries with corrective feedback
6. **Auto-review**: If enabled, runs LLM-based conversation review post-simulation
7. **Checkpoint save**: Saves completed simulation to disk for resume capability

```python
# batch.py (simplified)
def run_single_task(config, task, seed):
    orchestrator = build_orchestrator(config, task, seed=seed)
    simulation = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)
    return simulation
```

---

## Q4: How does `build_orchestrator()` construct the simulation components?

`build_orchestrator()` in `runner/build.py` dispatches to `build_text_orchestrator()` (for text mode):

### 1. Build Environment
```python
environment = build_environment(domain, env_kwargs=env_kwargs)
```
Uses the registry to get the domain's environment constructor. For retail:
```python
# domains/retail/environment.py
def get_environment():
    db = RetailDB(DB_PATH)
    tools = RetailTools(db)
    return Environment(domain_name="retail", tools=tools.get_tools(), policy=POLICY, ...)
```
The `Environment` wraps the database, agent tools (16 retail tools), user tools (if any), and the domain policy document.

### 2. Build Agent
```python
agent = build_agent(config.effective_agent, environment, llm=config.llm_agent, llm_args=config.llm_args_agent)
```
Resolves agent name via registry (default: `llm_agent`), gets factory function `create_llm_agent()`:
```python
# agent/llm_agent.py
def create_llm_agent(tools, domain_policy, llm, llm_args, ...):
    return LLMAgent(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
```
The `LLMAgent` stores the tools, policy (as system prompt), LLM model name, and args.

### 3. Build User
```python
user = build_user(config.effective_user, environment, task, llm=config.llm_user, llm_args=config.llm_args_user)
```
Creates a `UserSimulator` with:
- System prompt = global simulation guidelines + task-specific scenario
- The user-llm model and args
- Optional user tools (for domains like telecom where users also have tools)

### 4. Assemble Orchestrator
```python
orchestrator = Orchestrator(
    domain=domain, agent=agent, user=user, environment=environment,
    task=task, max_steps=200, max_errors=10, seed=seed, ...
)
```

---

## Q5: How does the Orchestrator run the simulation step-by-step?

The `Orchestrator.run()` method in `orchestrator/orchestrator.py`:

```python
def run(self):
    self.initialize()      # Set up initial state
    while not self.done:
        self.step()            # One turn of conversation
        self._check_termination()  # Check max_steps, max_errors, timeout
    return self._finalize()    # Wrap up, compute costs, return SimulationRun
```

### `initialize()`:
1. Initializes agent state: `self.agent_state = self.agent.get_init_state()`
   - LLMAgent creates system messages with domain policy
2. Initializes user state: `self.user_state = self.user.get_init_state()`
   - UserSimulator creates system messages with simulation guidelines + task scenario
3. Gets first user message: `user_msg, self.user_state = self.user.generate_next_message(init_msg, self.user_state)`
   - The user simulator LLM generates the opening customer message based on the task scenario
4. Sets routing: `from_role=USER, to_role=AGENT, message=user_msg`

### `step()` — the core loop:
Each step routes a message between three roles: AGENT, USER, ENV.

```
USER -> AGENT:
    agent_msg = agent.generate_next_message(user_msg, agent_state)
    if agent_msg is tool_call:  to_role = ENV
    else:                        to_role = USER

AGENT -> ENV:
    tool_results = environment.execute_tool_calls(agent_msg.tool_calls)
    to_role = AGENT  (results go back to agent)

ENV -> AGENT:
    (same as USER -> AGENT, but input is tool results)

AGENT -> USER:
    user_msg = user.generate_next_message(agent_msg, user_state)
    if user_msg is stop:  done = True
    if user_msg is tool_call:  to_role = ENV
    else:                       to_role = AGENT
```

The routing logic (from `orchestrator.py:823-899`):
```python
def step(self):
    # AGENT/ENV -> USER
    if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
        user_msg, self.user_state = self.user.generate_next_message(self.message, self.user_state)
        if UserSimulator.is_stop(user_msg):
            self.done = True; self.termination_reason = TerminationReason.USER_STOP
        self.trajectory.append(user_msg)
        self.message = user_msg
        self.from_role = Role.USER
        self.to_role = Role.AGENT if not user_msg.is_tool_call() else Role.ENV

    # USER/ENV -> AGENT
    elif self.from_role in [Role.USER, Role.ENV] and self.to_role == Role.AGENT:
        agent_msg, self.agent_state = self.agent.generate_next_message(self.message, self.agent_state)
        if self.agent.is_stop(agent_msg):
            self.done = True; self.termination_reason = TerminationReason.AGENT_STOP
        self.trajectory.append(agent_msg)
        self.message = agent_msg
        self.from_role = Role.AGENT
        self.to_role = Role.USER if not agent_msg.is_tool_call() else Role.ENV

    # AGENT/USER -> ENV
    elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
        tool_results = self._execute_tool_calls(self.message.tool_calls)
        self.trajectory.extend(tool_results)
        self.message = self._wrap_tool_results(tool_results)
        self.to_role = self.from_role  # results go back to whoever called
        self.from_role = Role.ENV
```

---

## Q6: How does the LLMAgent generate responses?

`LLMAgent.generate_next_message()` in `agent/llm_agent.py`:

```python
def generate_next_message(self, message, state):
    # Add incoming message to state
    state.messages.append(message)
    # Build messages = system_messages + conversation_history
    messages = state.system_messages + state.messages
    # Call LLM
    assistant_message = generate(
        model=self.llm,
        messages=messages,
        tools=self.tools,
        call_name="agent_response",
        **self.llm_args,
    )
    # Validate tool calls exist in available tools
    if assistant_message.is_tool_call():
        for tc in assistant_message.tool_calls:
            if tc.name not in [t.name for t in self.tools]:
                raise AgentError(f"Tool {tc.name} not found")
    state.messages.append(assistant_message)
    return assistant_message, state
```

The system prompt includes the domain policy document, which for retail contains rules like "always verify customer identity before making changes."

The `generate()` function in `utils/llm_utils.py` calls `litellm.completion()`:
```python
response = completion(
    model=model,           # e.g., "openrouter/qwen/qwen3-30b-a3b-instruct-2507"
    messages=litellm_messages,
    tools=tools_schema,    # OpenAI function calling format
    tool_choice="auto",
    **kwargs,              # temperature, api_base, api_key, etc.
)
```

---

## Q7: How does the UserSimulator generate responses?

`UserSimulator._generate_next_message()` in `user/user_simulator.py`:

```python
def _generate_next_message(self, message, state):
    # Add message to state
    state.messages.append(message)
    # IMPORTANT: flip_roles() swaps user<->assistant so the LLM sees itself as "assistant"
    messages = state.system_messages + state.flip_roles()
    # Generate response
    assistant_message = generate(
        model=self.llm,
        messages=messages,
        tools=self.tools,
        call_name="user_simulator_response",
        **self.llm_args,
    )
    # Convert assistant message to user message
    user_message = UserMessage(role="user", content=assistant_message.content, ...)
    return user_message
```

Key detail: `state.flip_roles()` swaps user/assistant roles because the user simulator is also an LLM generating "assistant" responses, but those responses should appear as "user" messages in the conversation. The user sees:
- System: simulation guidelines + task scenario (e.g., "You are a customer who wants to cancel order #W123")
- Previous conversation with roles flipped

The user simulator checks for stop signals:
```python
STOP = "[STOP]"
TRANSFER = "[TRANSFER]"
OUT_OF_SCOPE = "[OUT OF SCOPE]"
```
If the user message contains any of these, the orchestrator sets `done=True`.

---

## Q8: How does tool execution work in the Environment?

When the agent (or user) makes tool calls, the orchestrator routes to ENV:

```python
# orchestrator.py:882
tool_results = self._execute_tool_calls(self.message.tool_calls)
```

This calls `environment.use_tool()` for each tool call:
```python
# environment/environment.py
def use_tool(self, tool_call):
    tool_name = tool_call.name
    tool = self._tools_dict[tool_name]
    try:
        result = tool.function(**tool_call.arguments)
        return ToolMessage(content=json.dumps(result), error=False, ...)
    except Exception as e:
        return ToolMessage(content=str(e), error=True, ...)
```

For retail, tools like `get_order_details(order_id)` directly query the `RetailDB`:
```python
# domains/retail/tools.py
def get_order_details(self, order_id: str) -> dict:
    order = self.db.orders.get(order_id)
    if order is None:
        return {"error": f"Order {order_id} not found"}
    return order.model_dump()
```

Write tools like `cancel_pending_order(order_id, reason)` modify the database:
```python
def cancel_pending_order(self, order_id: str, reason: str) -> dict:
    order = self.db.orders[order_id]
    if order.status != "pending":
        return {"error": "Only pending orders can be cancelled"}
    order.status = "cancelled"
    order.cancel_reason = reason
    # Process refund...
    return {"success": True}
```

---

## Q9: How is the simulation evaluated after it completes?

After `orchestrator.run()` returns, `run_simulation()` in `runner/simulation.py` evaluates:

```python
def run_simulation(orchestrator):
    simulation = orchestrator.run()
    reward_info = evaluate_simulation(simulation, task, evaluation_type=...)
    simulation.reward_info = reward_info
    return simulation
```

`evaluate_simulation()` in `evaluator/evaluator.py` computes multiple reward components based on the task's `reward_basis`:

### Reward types:
1. **DB Match** (`RewardType.DB`): Compares final database state against expected state
   ```python
   db_reward = environment.check_db(task.expected_db_state)  # 1.0 or 0.0
   ```

2. **Environment Assertions** (`RewardType.ENV_ASSERTION`): Checks specific DB conditions
   ```python
   for assertion in task.evaluation_criteria.env_assertions:
       result = environment.check_assertion(assertion)
   ```

3. **Action Checks** (`RewardType.ACTION`): Verifies correct tool calls were made
   ```python
   action_reward = check_actions(trajectory, task.expected_actions)
   # Checks: right tool called? right arguments? right order?
   ```

4. **Communication Checks** (`RewardType.COMMUNICATE`): Verifies agent communicated required info
   ```python
   communicate_reward = check_communicate(trajectory, task.communicate_checks)
   ```

5. **NL Assertions** (`RewardType.NL_ASSERTION`): LLM-judged natural language assertions
   ```python
   # Sends full trajectory + assertions to an LLM (default: gpt-4.1) to judge
   nl_reward = NLAssertionsEvaluator.evaluate(simulation, task)
   ```

### Reward combination:
The final reward is a **product** of all applicable components (based on task's `reward_basis`):
```python
reward = db_reward * action_reward * communicate_reward * nl_reward
```
So if any component is 0, the total reward is 0.

---

## Q10: What does the output look like? How are results saved?

### Per-simulation output:
```
data/simulations/<run_name>/
  results.json                    # Aggregate: avg reward, pass^k, action metrics
  tasks/
    task_<id>/
      sim_<uuid>/
        sim_status.json          # Per-sim: reward, termination reason, eval details
        task.log                 # Human-readable conversation transcript
        llm_debug/               # Raw LLM request/response JSONs
          <timestamp>_agent_response_<id>.json
          <timestamp>_user_simulator_response_<id>.json
          <timestamp>_nl_assertions_eval_<id>.json
```

### `results.json` structure:
```json
{
  "info": { "agent_info": {...}, "user_info": {...}, "environment_info": {...} },
  "simulations": [
    {
      "task_id": "retail_task_1",
      "reward_info": { "reward": 1.0, "db_reward": 1.0, "action_reward": 1.0, ... },
      "termination_reason": "user_stop",
      "messages": [...],
      "duration": 8.5
    }
  ],
  "metrics": { "average_reward": 0.4, "pass_1": 0.4 }
}
```

### `sim_status.json` contains:
- Reward breakdown (DB, action, communicate, NL assertions)
- Per-action check results (which tool calls were correct/incorrect)
- NL assertion judgments with explanations
- Termination reason
- Cost and token usage
