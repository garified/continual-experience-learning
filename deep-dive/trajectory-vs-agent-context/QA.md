# Deep Dive: trajectory.md vs Actual Agent LLM Context

## Files Examined

- **Converter**: `/sfs/weka/scratch/ks8vf/exp/tau2-bench/scripts/trajectory_to_md.py`
- **Agent debug JSON**: `.../sim_2c1ee543.../llm_debug/20260319_183348_012_agent_response_30d4299b.json`
- **trajectory.md**: `.../tasks/task_0/trajectory.md`
- **Agent code**: `/sfs/weka/scratch/ks8vf/exp/tau2-bench/src/tau2/agent/llm_agent.py`
- **LLM utils**: `/sfs/weka/scratch/ks8vf/exp/tau2-bench/src/tau2/utils/llm_utils.py`
- **Orchestrator**: `/sfs/weka/scratch/ks8vf/exp/tau2-bench/src/tau2/orchestrator/orchestrator.py`
- **Data model**: `/sfs/weka/scratch/ks8vf/exp/tau2-bench/src/tau2/data_model/message.py`

---

## Q1: What is the exact structure of the raw agent debug JSON?

The debug JSON has this top-level structure:

```json
{
  "call_id": "30d4299b",
  "call_name": "agent_response",
  "timestamp": "2026-03-19T18:33:48.012873",
  "request": {
    "model": "openrouter/qwen/qwen3-30b-a3b-instruct-2507",
    "messages": [...],
    "tools": [...],
    "tool_choice": "auto",
    "kwargs": { "temperature": 0.0, "seed": 626729, "num_retries": 3 },
    "timestamp": "2026-03-19T18:33:45.790352"
  },
  "response": {
    "timestamp": "2026-03-19T18:33:48.010958",
    "content": "...",
    "tool_calls": null,
    "cost": 0.0,
    "usage": { "completion_tokens": 116, "prompt_tokens": 9370 },
    "generation_time_seconds": 2.219703009352088
  }
}
```

Key: `request.messages` is the actual array sent to litellm's `completion()`, and `request.tools` is the OpenAI-format tool definitions. The `response` captures the parsed output.

---

## Q2: How does `_format_messages_for_logging` transform messages before logging?

In `llm_utils.py` lines 270-289, the `_format_messages_for_logging` function splits string `content` fields on `\n` into arrays:

```python
if "content" in msg_copy and isinstance(msg_copy["content"], str):
    content_lines = msg_copy["content"].split("\n")
    if len(content_lines) > 1:
        msg_copy["content"] = content_lines
```

**CRITICAL FINDING**: This means the logged JSON does NOT faithfully represent what was sent to the API. Multi-line strings are converted to arrays of strings in the log.

Evidence from the debug JSON - the system message content is logged as an array:
```json
{
  "role": "system",
  "content": [
    "<instructions>",
    "You are a customer service agent...",
    ...
  ]
}
```

But what was actually sent to litellm was a single string with `\n` separators:
```
"<instructions>\nYou are a customer service agent..."
```

This applies to ALL messages with multi-line content: system, user, and assistant messages.

---

## Q3: What is the exact message format sent to the API?

The `to_litellm_messages()` function (llm_utils.py lines 168-208) converts Tau2 message types to litellm dicts:

### System messages:
```json
{"role": "system", "content": "<string>"}
```

### User messages:
```json
{"role": "user", "content": "<string>"}
```
Note: Only `content` is included. No `tool_calls` field is serialized for user messages, even though the Tau2 `UserMessage` model supports tool_calls.

### Assistant messages:
```json
{
  "role": "assistant",
  "content": "<string or null>",
  "tool_calls": [
    {
      "id": "chatcmpl-tool-d0fe0d9cf3d04a70a25a8558f3ea32f5",
      "name": "find_user_id_by_name_zip",
      "function": {
        "name": "find_user_id_by_name_zip",
        "arguments": "{\"first_name\": \"Yusuf\", \"last_name\": \"Rossi\", \"zip\": \"19122\"}"
      },
      "type": "function"
    }
  ]
}
```

Key observations:
- `tool_calls[].id` is present (unique identifier for correlating with tool results)
- `tool_calls[].name` is at the top level (redundant with `function.name`)
- `tool_calls[].function.arguments` is a **JSON string**, not a parsed object
- `tool_calls[].type` is always `"function"`
- `content` is `null` when there are tool calls (not omitted, explicitly null)
- `tool_calls` is `null` when there is no tool call (not omitted, explicitly null)

### Tool messages:
```json
{
  "role": "tool",
  "content": "<string>",
  "tool_call_id": "chatcmpl-tool-d0fe0d9cf3d04a70a25a8558f3ea32f5"
}
```

Key observations:
- Uses `tool_call_id`, NOT `id` (different from how it's stored in Tau2's ToolMessage which uses `id`)
- Does NOT include a `name` field (unlike some API formats)
- Content is a raw string (often a JSON string, but sent as plain string)

---

## Q4: What does trajectory.md drop or reformat vs the raw JSON?

### 4a. System prompt content format

**JSON (logged)**: Array of strings (one per line)
**JSON (actual API)**: Single string with `\n` newlines
**trajectory.md**: Single string with newlines (matches actual API, NOT the logged format)

The `format_content()` function in trajectory_to_md.py handles the array case by joining with `\n`:
```python
if isinstance(content, list):
    return "\n".join(str(c) for c in content)
```

**Verdict**: trajectory.md correctly reconstructs the original string. No discrepancy here.

### 4b. Tool call structure -- MAJOR DISCREPANCY

**Raw JSON**:
```json
{
  "id": "chatcmpl-tool-d0fe0d9cf3d04a70a25a8558f3ea32f5",
  "name": "find_user_id_by_name_zip",
  "function": {
    "name": "find_user_id_by_name_zip",
    "arguments": "{\"first_name\": \"Yusuf\", ...}"
  },
  "type": "function"
}
```

**trajectory.md renders as**:
```
### Assistant -> Tool Call

find_user_id_by_name_zip({
  "first_name": "Yusuf",
  "last_name": "Rossi",
  "zip": "19122"
})
```

**Dropped fields**:
1. `id` -- The tool_call ID is completely dropped. This is needed to correlate with tool results.
2. `type` -- The `"function"` type field is dropped.
3. `name` (top-level) -- Technically present in rendered form, but the redundant top-level `name` is merged with `function.name`.
4. `function.arguments` -- The JSON string is parsed and pretty-printed. The original was a raw JSON string; the md renders it as a formatted object inside a function-call-like syntax `name({...})`.

### 4c. Tool result structure -- MAJOR DISCREPANCY

**Raw JSON**:
```json
{
  "role": "tool",
  "content": "yusuf_rossi_9620",
  "tool_call_id": "chatcmpl-tool-d0fe0d9cf3d04a70a25a8558f3ea32f5"
}
```

**trajectory.md renders as**:
```
### Tool Result

yusuf_rossi_9620
```

**Dropped fields**:
1. `tool_call_id` -- Completely dropped. Cannot correlate which tool call this result belongs to.
2. `role` -- Implicit from the "### Tool Result" header, not in original API format.
3. No `name` field in the API format (litellm does not include one), but this is not a discrepancy.

### 4d. Assistant message with null content

**Raw JSON**:
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [...]
}
```

**trajectory.md**: The `null` content is correctly omitted (line 74 checks `if content.strip()`).

**Verdict**: Correct behavior.

### 4e. Assistant message with array content

**Raw JSON (logged)**:
```json
{
  "role": "assistant",
  "content": [
    "I found the following available options...",
    "",
    "1. **Mechanical Keyboard**:",
    ...
  ],
  "tool_calls": null
}
```

**Actual API value**: Single string with `\n` separators.

**trajectory.md**: Uses `format_content()` which joins arrays with `\n`. Result matches the original string.

**Verdict**: Correct reconstruction.

### 4f. `tools` array -- COMPLETELY MISSING FROM trajectory.md

The raw JSON includes `request.tools` which is a full array of 15 tool definitions in OpenAI schema format. This is part of the LLM request and directly impacts what the model can generate.

**trajectory.md**: Does not render the `tools` array at all.

This is a major omission for SFT because:
- The tools definition is part of the prompt context
- LLM tokenizers encode tools specially (as part of the prompt template)
- Without tools, the model does not know what functions are available

### 4g. `tool_choice` -- MISSING

The raw JSON includes `"tool_choice": "auto"`. This is not rendered in trajectory.md.

### 4h. Markdown formatting overhead

trajectory.md adds significant formatting that was NOT in the original context:
- `# Agent Trajectory` header
- `## Metadata` section with model/timestamp/call_id/temperature
- `---` horizontal rules between each message
- `### User`, `### Assistant`, `### Assistant -> Tool Call`, `### Tool Result`, `## Final Agent Response` headers
- Code block fences (` ``` ` and ` ```json `)
- `*Tokens: 9370 prompt, 116 completion*` footer
- `## Evaluation Results` section with reward info

None of these exist in the actual LLM context.

---

## Q5: Does the debug JSON contain ALL previous turns?

**YES.** The `request.messages` array in the final agent debug JSON contains the complete conversation history from the start:

1. System prompt
2. Assistant: "Hi! How can I help you today?" (the default first message)
3. User: "I'd like to exchange two items..."
4. Assistant: tool_call find_user_id_by_name_zip
5. Tool: "yusuf_rossi_9620"
6. Assistant: tool_call get_order_details
7. Tool: {order details JSON}
8. Assistant: tool_call get_product_details (keyboard)
9. Tool: {keyboard variants JSON}
10. Assistant: tool_call get_product_details (thermostat)
11. Tool: {thermostat variants JSON}
12. Assistant: "I found the following available options..."
13. User: "Yes, I'd like to proceed..."
14. Assistant: tool_call exchange_delivered_order_items
15. Tool: {exchange result JSON}
16. Assistant: "Your exchange request has been successfully processed!"
17. User: "Thank you for confirming..."

The response is the FINAL agent turn (turn 18):
```
"Yes, once the return is processed..."
```

**This means we CAN reconstruct all per-turn training examples from the single final debug JSON**, because each assistant turn in the messages array was a completion at that step, and everything before it was the context.

---

## Q6: Can we reconstruct per-turn (context, completion) pairs?

**YES, with caveats.** By iterating through the messages array, we can find each assistant message and treat:
- **Context**: system message + all messages before this assistant message + tools
- **Completion**: this assistant message

However, there are important caveats:

### Caveat 1: The `tools` array is constant across all turns
The same tools definition would be used for each turn's context. This is correct since the agent uses the same tools throughout.

### Caveat 2: `_format_messages_for_logging` corrupts multi-line strings
Multi-line strings are split into arrays. To reconstruct the exact API input, we must:
- If `content` is an array, join with `\n`
- If `content` is a string, use as-is

### Caveat 3: The first assistant message is a hardcoded default
```python
DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)
```
This is injected by the Orchestrator, NOT generated by the LLM. It should be treated as context, not as a completion to train on.

### Caveat 4: kwargs differ between logged and actual
The logged kwargs include `num_retries: 3` which is a litellm parameter, not an LLM parameter. The actual API call may have different parameters.

---

## Q7: What is the exact token-level difference between trajectory.md and the actual API?

For SFT, the tokenizer processes the messages array using a chat template. The actual token sequence looks like (for a typical model):

```
<|im_start|>system
<instructions>
You are a customer service agent...
</instructions>
<policy>
...
</policy><|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
<|im_start|>user
I'd like to exchange...<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "find_user_id_by_name_zip", "arguments": {...}}
</tool_call><|im_end|>
<|im_start|>tool
yusuf_rossi_9620<|im_end|>
...
```

The markdown rendering adds hundreds of tokens that don't exist:
- `# Agent Trajectory\n\n## Metadata\n\n- **Model**: ...`
- `---\n## System Prompt\n\n`
- `---\n### User\n\n`
- `---\n### Assistant -> Tool Call\n\n```\n`
- etc.

These tokens would train the model to expect and generate markdown formatting, which is completely wrong.

---

## Q8: How does the log mode affect what's available?

From `llm_utils.py` lines 257-267:

```python
llm_log_mode: ContextVar[str] = ContextVar("llm_log_mode", default="latest")
```

Default mode is `"latest"`, which means:
- Only the MOST RECENT LLM call of each `call_name` is kept
- Previous calls with the same `call_name` are deleted (lines 320-329)

For the agent, `call_name="agent_response"`, so only the final agent turn's debug JSON exists. Earlier agent turns are overwritten.

**Impact on SFT**: Since all previous turns are included in the final turn's `request.messages`, we CAN still reconstruct earlier turns. But we lose the individual `response` objects for earlier turns (their usage stats, generation time, etc.).

---

## Q9: What about the user simulator debug JSON?

The user simulator debug JSON shows a completely DIFFERENT message format:
- Roles are SWAPPED: agent messages appear as `role: "user"` and user messages as `role: "assistant"` (from the user sim's perspective)
- The system prompt is the user simulator instructions (scenario)
- No tools array (user sim doesn't call tools in this case)
- Content is also affected by `_format_messages_for_logging` (multi-line strings become arrays)

**This is NOT relevant for agent SFT** -- we only need the agent's perspective.

---

## Q10: What about the `name` field on tool messages?

In the OpenAI API format, tool messages can optionally include a `name` field. Looking at `to_litellm_messages()`:

```python
elif isinstance(message, ToolMessage):
    litellm_messages.append(
        {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.id,
        }
    )
```

The Tau2 code does NOT include a `name` field on tool messages. The logged JSON confirms this -- tool messages only have `role`, `content`, and `tool_call_id`.

---

## Q11: How are tool_call arguments serialized?

In `to_litellm_messages()`:

```python
"function": {
    "name": tc.name,
    "arguments": json.dumps(tc.arguments),
}
```

The `arguments` field is `json.dumps(tc.arguments)` -- a JSON STRING, not a dict. This is critical for SFT because the tokenizer must see the exact string representation.

In the debug JSON, arguments appear as a JSON string:
```json
"arguments": "{\"first_name\": \"Yusuf\", \"last_name\": \"Rossi\", \"zip\": \"19122\"}"
```

In trajectory.md, this is parsed back into a dict and pretty-printed:
```
find_user_id_by_name_zip({
  "first_name": "Yusuf",
  "last_name": "Rossi",
  "zip": "19122"
})
```

**Discrepancy**: The original is compact JSON string, the md shows pretty-printed inside a function-call syntax.

---

## Q12: What response fields are in the debug JSON but not in trajectory.md?

The `response` object includes:
- `timestamp` -- not in md (well, included in metadata as request timestamp)
- `content` -- rendered as "## Final Agent Response"
- `tool_calls` -- would be rendered if present
- `cost` -- not rendered
- `usage` -- rendered as `*Tokens: 9370 prompt, 116 completion*`
- `generation_time_seconds` -- not rendered

---

## Q13: Is the `tool_calls` in the response the same format as in messages?

The response `tool_calls` come from `llm_utils.py` lines 456-460:

```python
response_data = {
    ...
    "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
    ...
}
```

This is the Tau2 `ToolCall.model_dump()` format:
```json
{
  "id": "...",
  "name": "...",
  "arguments": {...},  // dict, NOT string
  "requestor": "assistant"
}
```

This is DIFFERENT from the litellm format in `request.messages`:
```json
{
  "id": "...",
  "name": "...",
  "function": { "name": "...", "arguments": "<json-string>" },
  "type": "function"
}
```

**Impact**: When reconstructing training data, the response tool_calls must be converted to litellm format. The messages already have litellm format (since they went through `to_litellm_messages`).

---

## Summary of ALL Discrepancies

| Aspect | Raw JSON (actual API) | trajectory.md | SFT Impact |
|--------|----------------------|---------------|------------|
| System content | Single string with `\n` | Same (after join) | Match - OK |
| User content | Single string with `\n` | Same (after join) | Match - OK |
| Assistant content | Single string or null | Same (after join) | Match - OK |
| Tool call `id` | Present | **DROPPED** | CRITICAL - needed for multi-tool correlation |
| Tool call `type` | `"function"` | **DROPPED** | Moderate - part of API format |
| Tool call `function.arguments` | JSON string | **Pretty-printed dict in `name(...)` syntax** | CRITICAL - format mismatch |
| Tool message `tool_call_id` | Present | **DROPPED** | CRITICAL - needed for correlation |
| Tool message `role` | `"tool"` | Implicit via header | CRITICAL - must be explicit for API |
| `tools` array | 15 tool definitions | **COMPLETELY MISSING** | CRITICAL - defines available functions |
| `tool_choice` | `"auto"` | **MISSING** | Moderate |
| Markdown headers/formatting | Not present | **ADDED** | CRITICAL - pollutes training data |
| Code fences | Not present | **ADDED** | CRITICAL - wrong tokens |
| Metadata section | Not present | **ADDED** | Must exclude from training |
| Eval results section | Not present | **ADDED** | Must exclude from training |
| Message ordering | Array order | Same | Match - OK |
| Content array -> string | Split by `\n` in log | Rejoined by `\n` | Match after reconstruction |
| Response tool_calls format | Tau2 ToolCall (dict args) | litellm format (string args) | Must normalize |
| kwargs (temperature, seed) | Present in request | Only temperature in metadata | Minor |
| `content: null` on tool-call msgs | Explicit null | Omitted (correct) | OK |
| `tool_calls: null` on text msgs | Explicit null | Omitted (correct) | OK for SFT |
