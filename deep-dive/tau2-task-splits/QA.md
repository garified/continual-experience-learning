# Deep Dive: tau2-bench Train/Test Task Splits

## Q1: Is there a clear separation of training and testing tasks in tau2-bench?

**Yes.** Every major domain (retail, airline, telecom) ships with a `split_tasks.json` file that defines disjoint `train` and `test` splits, plus a `base` split that is the union of both.

The split files live alongside the task data:
```
data/tau2/domains/retail/split_tasks.json
data/tau2/domains/airline/split_tasks.json
data/tau2/domains/telecom/split_tasks.json
```

### Split sizes:

| Domain   | Train | Test | Base (=Train+Test) | Other Splits |
|----------|-------|------|---------------------|--------------|
| Retail   | 74    | 40   | 114                 | —            |
| Airline  | 30    | 20   | 50                  | —            |
| Telecom  | 74    | 40   | 114 (base), 2285 (full) | small (20) |
| Mock     | —     | —    | 10                  | —            |
| Banking  | —     | —    | N/A (no splits)     | —            |

**Train and test are fully disjoint** — zero overlap in every domain. The `base` split equals `train ∪ test`.

---

## Q2: How are the splits stored on disk?

Each domain has a `split_tasks.json` file in its data directory. The format is a JSON dict mapping split names to arrays of task ID strings:

```json
// data/tau2/domains/retail/split_tasks.json
{
    "train": ["0", "1", "2", "3", "4", "6", "7", ...],   // 74 IDs
    "test":  ["5", "9", "12", "17", "18", ...],           // 40 IDs
    "base":  ["0", "1", "2", ..., "5", "9", "12", ...]    // 114 IDs (train then test)
}
```

The split file is resolved relative to the task data file. For retail:

```python
# src/tau2/domains/retail/environment.py:49-54
def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(RETAIL_TASK_SET_PATH).parent
        / f"split_{Path(RETAIL_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
```

So if `RETAIL_TASK_SET_PATH` is `data/tau2/domains/retail/tasks.json`, the split file is `data/tau2/domains/retail/split_tasks.json`.

---

## Q3: How does the code load tasks by split?

The loading chain is:

1. **CLI** (`cli.py:112-117`): `--task-split-name` arg, defaults to `"base"`:
   ```python
   parser.add_argument(
       "--task-split-name",
       type=str,
       default="base",
       help="The task split to run the simulation on. If not provided, will load 'base' split.",
   )
   ```

2. **Runner** (`runner/helpers.py:49-53`): `load_tasks()` passes the split name to the domain's task loader:
   ```python
   def load_tasks(task_set_name: str, task_split_name: Optional[str] = None) -> list[Task]:
       task_loader = registry.get_tasks_loader(task_set_name)
       tasks = task_loader(task_split_name=task_split_name)
       return tasks
   ```

3. **Domain loader** (e.g., `domains/retail/environment.py:35-46`): Loads all tasks, then filters by split:
   ```python
   def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
       tasks = load_file(RETAIL_TASK_SET_PATH)
       tasks = [Task.model_validate(task) for task in tasks]
       if task_split_name is None:
           return tasks  # ALL tasks, no filtering
       task_splits = get_tasks_split()
       if task_split_name not in task_splits:
           raise ValueError(f"Invalid task split name: {task_split_name}.")
       tasks = [task for task in tasks if task.id in task_splits[task_split_name]]
       return tasks
   ```

Passing `--task-split-name train` loads only the 74 train tasks. Passing `--task-split-name test` loads only the 40 test tasks.

---

## Q4: What's the default behavior? What did our earlier 5-task run use?

The default is `--task-split-name base`, which loads **all 114 tasks** (train + test combined). Our earlier run command was:

```bash
tau2 run --domain retail \
  --agent-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" \
  --user-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" \
  --num-tasks 5
```

This used the default `base` split and `--num-tasks 5` truncated to the first 5 tasks. Looking at the output:

```
data/simulations/.../tasks/
  task_0/  task_1/  task_2/  task_3/  task_4/
```

Task IDs 0-4 are all in the **train** split. The `base` split lists train IDs first, so `--num-tasks N` with small N will always pull train tasks first.

---

## Q5: Does the Task data model itself carry a split label?

**No.** The `Task` model (`data_model/tasks.py:495-557`) has no split field:

```python
class Task(BaseModel):
    id: str
    description: Optional[Description]
    user_scenario: UserScenario
    ticket: Optional[str]
    initial_state: Optional[InitialState]
    evaluation_criteria: Optional[EvaluationCriteria]
    issues: Optional[list[TaskIssue]]
    required_documents: Optional[list[str]]
    user_tools: Optional[list[str]]
```

Split membership is entirely external — defined by the `split_tasks.json` file and resolved at load time via the domain's `get_tasks()` function.

---

## Q6: How does the registry wire up split support?

The `Registry` class (`registry.py:82-83`) stores separate callables for tasks and splits:

```python
self._tasks: Dict[str, Callable[[Optional[str]], list[Task]]] = {}
self._task_splits: Dict[str, Callable[[], dict[str, list[str]]]] = {}
```

Registration happens at import time (`registry.py:310-322`):

```python
registry.register_tasks(
    retail_domain_get_tasks,
    "retail",
    get_task_splits=retail_domain_get_tasks_split,
)
```

The `register_tasks()` method (`registry.py:192-212`):
```python
def register_tasks(
    self,
    get_tasks: Callable[[Optional[str]], list[Task]],
    name: str,
    get_task_splits: Optional[Callable[[], dict[str, list[str]]]] = None,
):
```

---

## Q7: What does the telecom domain do differently?

Telecom has **5 splits** — more than retail/airline:

| Split | Count | Description |
|-------|-------|-------------|
| `small` | 20 | Single-issue scenarios (e.g., one troubleshooting step) |
| `train` | 74 | Multi-issue combinations for training |
| `test` | 40 | Multi-issue combinations for testing |
| `base` | 114 | train + test |
| `full` | 2,285 | All combinatorial scenarios |

The `small` split has simple, single-fault tasks like `[mobile_data_issue]user_abroad_roaming_enabled_off`. Train and test have multi-fault compound tasks like `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|...`.

Telecom also has legacy helper functions (`domains/telecom/environment.py:186-194`):
```python
def get_tasks_full() -> list[Task]:
    return get_tasks("full")

def get_tasks_small() -> list[Task]:
    return get_tasks("small")
```

---

## Q8: Which domains DON'T have train/test splits?

- **Mock**: Only has a `base` split (10 tasks). Used for integration testing, not benchmarking.
- **Banking Knowledge**: No split support at all. The `get_tasks()` function ignores the `task_split_name` parameter:
  ```python
  # domains/banking_knowledge/environment.py:95
  def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
      """Args:
          task_split_name: Optional task split name (not used for banking_knowledge domain yet)
      """
  ```

---

## Q9: Can you run on specific task IDs regardless of split?

Yes. The `--task-ids` CLI arg lets you specify exact task IDs:

```bash
tau2 run --domain retail --task-ids 5 9 12 17
```

The filtering logic in `runner/helpers.py:56-93`:
```python
def get_tasks(
    task_set_name: str,
    task_split_name: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    num_tasks: Optional[int] = None,
) -> list[Task]:
    tasks = load_tasks(task_set_name, task_split_name)
    if task_ids:
        tasks = [task for task in tasks if task.id in task_ids]
    if num_tasks:
        tasks = tasks[:num_tasks]
    return tasks
```

`--task-ids` is applied **after** split filtering. So `--task-split-name test --task-ids 5 9` would load only IDs 5 and 9 from the test split.

---

## Q10: What are the practical implications for our fine-tuning workflow?

This is important for our knowledge extraction fine-tuning work:

1. **For generating training data**: Use `--task-split-name train` to only run on train tasks. This gives us 74 retail tasks to generate SFT data from.

2. **For evaluation**: Use `--task-split-name test` to run on the held-out 40 test tasks. This ensures no data leakage.

3. **For benchmarking**: The default `--task-split-name base` (114 tasks) matches the official leaderboard evaluation. The tau2-bench leaderboard reports results on all base tasks.

4. **Our earlier 5-task run**: Used base split (default), which started with train IDs 0-4. To get a proper held-out evaluation, we should explicitly use `--task-split-name test`.

Commands for proper separation:
```bash
# Train: generate trajectories for SFT data
tau2 run --domain retail --task-split-name train \
  --agent-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" ...

# Eval: test on held-out tasks
tau2 run --domain retail --task-split-name test \
  --agent-llm "openrouter/qwen/qwen3-30b-a3b-instruct-2507" ...
```
