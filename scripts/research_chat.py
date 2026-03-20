"""
Gradio research chat interface for HotpotQA checkpoints.

Two tabs:
  - Chat: multi-turn conversation with any checkpoint
  - Compare: send the same prompt to multiple checkpoints, see responses side by side

Usage:
    python -u scripts/research_chat.py
    python -u scripts/research_chat.py --port 8888
    python -u scripts/research_chat.py --root-path "/node/host/6592/proxy/7861/"
    python -u scripts/research_chat.py --share
"""

import json
import asyncio
import argparse
import os
import re
import string
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
RUNS_DIR = "/sfs/weka/scratch/ks8vf/exp/runs"
DEFAULT_PORT = 7861
MAX_CACHED_CLIENTS = 10

# Lazily initialized renderer/tokenizer (deferred to avoid import-time errors
# in environments where transformers dependencies aren't fully available).
_renderer = None
_tokenizer = None


def _get_renderer():
    global _renderer, _tokenizer
    if _renderer is None:
        renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
        _tokenizer = get_tokenizer(MODEL_NAME)
        _renderer = get_renderer(renderer_name, _tokenizer)
    return _renderer


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
@dataclass
class CheckpointInfo:
    display_name: str  # e.g. "v2/000012 (ep0)" or "base_model"
    version: str       # e.g. "v2", "v5_s1"
    step: str          # e.g. "000012", "final"
    sampler_path: str  # tinker:// URI (empty for base model)
    epoch: int = 0
    is_base: bool = False


def discover_checkpoints() -> list[CheckpointInfo]:
    """Scan runs/hotpotqa_*/checkpoints.jsonl and runs/paper_cpt_*/checkpoints.jsonl."""
    checkpoints: list[CheckpointInfo] = []

    # Base model as special entry
    checkpoints.append(CheckpointInfo(
        display_name="base_model",
        version="base",
        step="base",
        sampler_path="",
        epoch=0,
        is_base=True,
    ))

    runs_path = Path(RUNS_DIR)

    # Patterns to scan
    run_patterns = [
        # ("hotpotqa_*", "hotpotqa_"),  # disabled: old HotpotQA experiments
        ("paper_cpt_*", "paper_cpt_"),
    ]

    for pattern, prefix in run_patterns:
        for run_dir in sorted(runs_path.glob(pattern)):
            ckpt_file = run_dir / "checkpoints.jsonl"
            if not ckpt_file.exists():
                continue

            # Extract version label: hotpotqa_v2 -> v2, paper_cpt_v12 -> cpt_v12
            version = run_dir.name.replace("hotpotqa_", "").replace("paper_cpt_", "cpt_")

            with open(ckpt_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    ckpt = json.loads(line)
                    epoch = ckpt.get("epoch", 0)
                    step_name = ckpt["name"]
                    display = f"{version}/{step_name} (ep{epoch})"

                    checkpoints.append(CheckpointInfo(
                        display_name=display,
                        version=version,
                        step=step_name,
                        sampler_path=ckpt["sampler_path"],
                        epoch=epoch,
                    ))

    return checkpoints


# ---------------------------------------------------------------------------
# Sampling client cache
# ---------------------------------------------------------------------------
class SamplingClientCache:
    """LRU cache of tinker.SamplingClient instances (max_size).

    Concurrent requests for the same key coalesce via asyncio.Task sharing.
    """

    def __init__(self, max_size: int = MAX_CACHED_CLIENTS):
        self._max_size = max_size
        self._cache: OrderedDict[str, tinker.SamplingClient] = OrderedDict()
        self._lock = asyncio.Lock()
        # In-flight creation tasks keyed by display_name
        self._inflight: dict[str, asyncio.Task] = {}

    async def get(self, info: CheckpointInfo) -> tinker.SamplingClient:
        key = info.display_name

        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            if key not in self._inflight:
                self._inflight[key] = asyncio.create_task(self._create_client(info))
            task = self._inflight[key]

        # Await outside lock so other coroutines can proceed
        try:
            client = await asyncio.shield(task)
        except Exception:
            async with self._lock:
                self._inflight.pop(key, None)
            raise

        async with self._lock:
            if key not in self._cache:
                self._cache[key] = client
                self._cache.move_to_end(key)
                while len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)
            self._inflight.pop(key, None)
            return self._cache[key]

    @staticmethod
    async def _create_client(info: CheckpointInfo) -> tinker.SamplingClient:
        service_client = tinker.ServiceClient()
        if info.is_base:
            return await service_client.create_sampling_client_async(base_model=MODEL_NAME)
        return await service_client.create_sampling_client_async(model_path=info.sampler_path)


# Global cache instance
_client_cache = SamplingClientCache()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
async def generate_response(
    info: CheckpointInfo,
    messages: list[dict],
    temperature: float = 0.6,
    max_tokens: int = 512,
) -> str:
    """Send messages to a checkpoint and return the response text."""
    sampling_client = await _client_cache.get(info)
    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=_get_renderer(),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    response = await completer(messages)
    return response["content"].strip()


# ---------------------------------------------------------------------------
# EM / F1 scoring (SQuAD-style)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens) if pred_tokens else 0
    recall = num_same / len(gold_tokens) if gold_tokens else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# ---------------------------------------------------------------------------
# Chat tab handler
# ---------------------------------------------------------------------------
async def chat_respond(
    message: str,
    history: list[dict],
    checkpoint_name: str,
    temperature: float,
    max_tokens: int,
    checkpoint_map: dict[str, CheckpointInfo],
):
    """Handle a chat message: append user msg, call model, yield assistant msg."""
    if not checkpoint_name:
        history.append({"role": "assistant", "content": "Please select a checkpoint first."})
        yield history
        return

    info = checkpoint_map.get(checkpoint_name)
    if info is None:
        history.append({"role": "assistant", "content": f"Checkpoint '{checkpoint_name}' not found."})
        yield history
        return

    # Build messages list from history + new user message
    messages = []
    for entry in history:
        messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": message})

    # Yield user message immediately
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "..."})
    yield history

    try:
        response = await generate_response(info, messages, temperature, int(max_tokens))
        history[-1] = {"role": "assistant", "content": response}
    except Exception as e:
        history[-1] = {"role": "assistant", "content": f"Error: {e}"}
    yield history


# ---------------------------------------------------------------------------
# Compare tab handlers
# ---------------------------------------------------------------------------
async def compare_handler(
    prompt: str,
    gold_answer: str,
    selected_checkpoints: list[str],
    temperature: float,
    max_tokens: int,
    checkpoint_map: dict[str, CheckpointInfo],
):
    """Send the same prompt to multiple checkpoints concurrently."""
    if not prompt.strip():
        return []
    if not selected_checkpoints:
        return [["No checkpoints selected", "", "", "", ""]]

    semaphore = asyncio.Semaphore(20)

    async def run_one(name: str) -> list:
        info = checkpoint_map.get(name)
        if info is None:
            return [name, f"Not found", "", "", ""]
        async with semaphore:
            t0 = time.time()
            try:
                messages = [{"role": "user", "content": prompt}]
                resp = await generate_response(info, messages, temperature, int(max_tokens))
                latency = time.time() - t0
            except Exception as e:
                return [name, f"Error: {e}", "", "", ""]

        em_val, f1_val = "", ""
        if gold_answer.strip():
            em_val = f"{exact_match(resp, gold_answer):.0f}"
            f1_val = f"{f1_score(resp, gold_answer):.2f}"
        return [name, resp, em_val, f1_val, f"{latency:.1f}s"]

    tasks = [run_one(n) for n in selected_checkpoints]
    results = await asyncio.gather(*tasks)
    return list(results)


async def batch_compare_handler(
    jsonl_text: str,
    selected_checkpoints: list[str],
    temperature: float,
    max_tokens: int,
    checkpoint_map: dict[str, CheckpointInfo],
):
    """Run batch comparison: each JSONL line x each checkpoint."""
    if not jsonl_text.strip():
        return [["No input provided", "", "", "", "", "", ""]]
    if not selected_checkpoints:
        return [["", "", "No checkpoints selected", "", "", "", ""]]

    # Parse JSONL lines
    lines = []
    for line in jsonl_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            lines.append(obj)
        except json.JSONDecodeError as e:
            return [[f"JSON parse error: {e}", "", "", "", "", "", ""]]

    if not lines:
        return [["No valid JSONL lines", "", "", "", "", "", ""]]

    semaphore = asyncio.Semaphore(20)

    async def run_one(prompt: str, gold: str, ckpt_name: str) -> list:
        info = checkpoint_map.get(ckpt_name)
        if info is None:
            return [prompt[:80], gold, ckpt_name, "Not found", "", "", ""]
        async with semaphore:
            t0 = time.time()
            try:
                messages = [{"role": "user", "content": prompt}]
                resp = await generate_response(info, messages, temperature, int(max_tokens))
                latency = time.time() - t0
            except Exception as e:
                return [prompt[:80], gold, ckpt_name, f"Error: {e}", "", "", ""]

        em_val, f1_val = "", ""
        if gold.strip():
            em_val = f"{exact_match(resp, gold):.0f}"
            f1_val = f"{f1_score(resp, gold):.2f}"
        return [prompt[:80], gold, ckpt_name, resp, em_val, f1_val, f"{latency:.1f}s"]

    tasks = []
    for item in lines:
        prompt = item.get("prompt", item.get("question", ""))
        gold = item.get("gold_answer", item.get("answer", ""))
        for ckpt_name in selected_checkpoints:
            tasks.append(run_one(prompt, gold, ckpt_name))

    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# OOD proxy auto-detection
# ---------------------------------------------------------------------------
def detect_ood_root_path(port: int) -> str | None:
    """Try to auto-detect Open OnDemand proxy root_path."""
    hostname = os.environ.get("HOSTNAME", "")
    if not hostname:
        return None

    # Look for Jupyter port from common env vars
    jupyter_port = os.environ.get("JUPYTER_PORT", "")
    if not jupyter_port:
        # Try to infer from JUPYTERHUB_SERVICE_PREFIX or runtime dir
        prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "")
        if prefix:
            # e.g. /user/ks8vf/ — doesn't directly give port
            pass
        # Check for jupyter runtime files that might indicate the port
        runtime_dir = os.environ.get("JUPYTER_RUNTIME_DIR", "")
        if runtime_dir:
            import glob
            json_files = sorted(Path(runtime_dir).glob("*server-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            for jf in json_files:
                try:
                    with open(jf) as f:
                        info = json.load(f)
                    jupyter_port = str(info.get("port", ""))
                    if jupyter_port:
                        break
                except Exception:
                    continue

    if not jupyter_port:
        return None

    root_path = f"/node/{hostname}/{jupyter_port}/proxy/{port}/"
    return root_path


# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------
def build_interface(checkpoints: list[CheckpointInfo]) -> gr.Blocks:
    """Construct the Gradio Blocks UI."""
    checkpoint_map = {c.display_name: c for c in checkpoints}
    display_names = [c.display_name for c in checkpoints]

    with gr.Blocks(title="HotpotQA Research Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# HotpotQA Research Chat\n"
                     f"**{len(checkpoints)} checkpoints** available "
                     f"(model: `{MODEL_NAME}`)")

        # ---- Chat Tab ----
        with gr.Tab("Chat"):
            with gr.Row():
                chat_ckpt = gr.Dropdown(
                    choices=display_names,
                    value="base_model",
                    label="Checkpoint",
                    filterable=True,
                    scale=3,
                )
                chat_temp = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.6, step=0.05,
                    label="Temperature", scale=1,
                )
                chat_max_tok = gr.Slider(
                    minimum=16, maximum=4096, value=512, step=16,
                    label="Max Tokens", scale=1,
                )

            chatbot = gr.Chatbot(type="messages", height=500)
            msg_input = gr.Textbox(
                placeholder="Type your message...",
                label="Message",
                lines=2,
            )
            clear_btn = gr.Button("Clear Chat")

            async def handle_chat(message, history, ckpt_name, temp, max_tok):
                async for updated in chat_respond(
                    message, history, ckpt_name, temp, max_tok, checkpoint_map
                ):
                    yield updated, ""

            msg_input.submit(
                handle_chat,
                inputs=[msg_input, chatbot, chat_ckpt, chat_temp, chat_max_tok],
                outputs=[chatbot, msg_input],
            )
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

        # ---- Compare Tab ----
        with gr.Tab("Compare"):
            gr.Markdown("### Single Prompt Comparison")
            with gr.Row():
                cmp_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter a question...",
                    lines=3,
                    scale=2,
                )
                cmp_gold = gr.Textbox(
                    label="Gold Answer (optional)",
                    placeholder="Expected answer for EM/F1",
                    lines=1,
                    scale=1,
                )

            cmp_ckpts = gr.Dropdown(
                choices=display_names,
                value=["base_model"],
                label="Checkpoints to compare",
                multiselect=True,
                filterable=True,
            )
            with gr.Row():
                cmp_temp = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.6, step=0.05,
                    label="Temperature",
                )
                cmp_max_tok = gr.Slider(
                    minimum=16, maximum=4096, value=512, step=16,
                    label="Max Tokens",
                )
                cmp_run_btn = gr.Button("Run Comparison", variant="primary")

            cmp_results = gr.Dataframe(
                headers=["Checkpoint", "Response", "EM", "F1", "Latency"],
                column_widths=["20%", "45%", "8%", "8%", "10%"],
                wrap=True,
            )

            async def handle_compare(prompt, gold, ckpts, temp, max_tok):
                return await compare_handler(
                    prompt, gold, ckpts, temp, max_tok, checkpoint_map
                )

            cmp_run_btn.click(
                handle_compare,
                inputs=[cmp_prompt, cmp_gold, cmp_ckpts, cmp_temp, cmp_max_tok],
                outputs=[cmp_results],
            )

            # ---- Batch sub-section ----
            gr.Markdown("---\n### Batch Comparison")
            batch_input = gr.Textbox(
                label="Batch JSONL Input",
                placeholder='{"prompt": "What is...", "gold_answer": "..."}\n{"prompt": "Who was...", "gold_answer": "..."}',
                lines=6,
            )
            batch_run_btn = gr.Button("Run Batch", variant="primary")
            batch_results = gr.Dataframe(
                headers=["Prompt", "Gold", "Checkpoint", "Response", "EM", "F1", "Latency"],
                column_widths=["15%", "10%", "15%", "35%", "6%", "6%", "8%"],
                wrap=True,
            )

            async def handle_batch(jsonl_text, ckpts, temp, max_tok):
                return await batch_compare_handler(
                    jsonl_text, ckpts, temp, max_tok, checkpoint_map
                )

            batch_run_btn.click(
                handle_batch,
                inputs=[batch_input, cmp_ckpts, cmp_temp, cmp_max_tok],
                outputs=[batch_results],
            )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HotpotQA Research Chat Interface")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to serve on")
    parser.add_argument("--root-path", type=str, default=None,
                        help="Root path for reverse proxy (auto-detected for OOD)")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    # Discover checkpoints
    print("Discovering checkpoints...")
    checkpoints = discover_checkpoints()
    print(f"Found {len(checkpoints)} checkpoints (including base_model)")

    # Print version breakdown
    version_counts = Counter(c.version for c in checkpoints if not c.is_base)
    for v, cnt in sorted(version_counts.items()):
        print(f"  {v}: {cnt} checkpoints")

    # Detect or use root_path
    root_path = args.root_path
    if root_path is None and not args.share:
        root_path = detect_ood_root_path(args.port)
        if root_path:
            print(f"Auto-detected OOD root_path: {root_path}")

    # Build URL for user
    hostname = os.environ.get("HOSTNAME", "localhost")
    if root_path:
        print(f"\nAccess via OOD proxy:")
        print(f"  https://ood.hpc.virginia.edu{root_path}")
    else:
        print(f"\nAccess at: http://{hostname}:{args.port}")

    # Build and launch
    demo = build_interface(checkpoints)

    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": args.share,
    }
    if root_path:
        launch_kwargs["root_path"] = root_path

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
