"""
Prepare HotpotQA training data v4: All 100 questions (same as v2 but scaled up).

Type 1: Passage absorption (chat pairs from ctxs)
Type 2: Multi-hop QA generation (teacher discovers links, generates questions)

Uses OpenRouter for teacher model with parallel inference.
"""

import json
import asyncio
import os
import random
from pathlib import Path
from openai import AsyncOpenAI

INPUT_FILE = "/sfs/weka/scratch/ks8vf/exp/HELMET/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl"
OUTPUT_DIR = Path("/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_100")
TEACHER_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
NUM_QUESTIONS = 100  # All 100 questions
MAX_CONCURRENT = 10


def load_first_n_unique_questions(filepath: str, n: int) -> list[dict]:
    """Load first N unique questions from HELMET HotpotQA file."""
    with open(filepath) as f:
        data = [json.loads(line) for line in f]

    seen = set()
    result = []
    for d in data:
        if d['question'] not in seen and len(seen) < n:
            seen.add(d['question'])
            result.append(d)
    return result


def create_type1_data(samples: list[dict]) -> list[dict]:
    """
    Type 1: Convert passages to chat format for knowledge absorption.
    Format: user asks about title, assistant provides passage text.
    """
    type1_data = []
    seen_passages = set()

    for sample in samples:
        for ctx in sample['ctxs']:
            psg_id = ctx.get('id', ctx['title'])
            if psg_id in seen_passages:
                continue
            seen_passages.add(psg_id)

            type1_data.append({
                "messages": [
                    {"role": "user", "content": f"Tell me about {ctx['title']}"},
                    {"role": "assistant", "content": ctx['text']}
                ]
            })

    return type1_data


LINK_DISCOVERY_PROMPT = """You are analyzing a set of documents to find pairs that share entities, topics, or facts that could support multi-hop reasoning questions.

Here are the documents:

{documents}

Task: Identify pairs of documents that share meaningful connections (shared people, events, places, concepts, or facts). For each pair, briefly explain the connection.

Output format (JSON array):
[
  {{"doc1_idx": 0, "doc2_idx": 3, "connection": "Both mention person X"}},
  {{"doc1_idx": 1, "doc2_idx": 5, "connection": "Doc1 describes event Y, Doc2 provides context about Y"}}
]

Only include pairs with genuine topical overlap that could support a question requiring both documents. Output valid JSON only."""


QA_GENERATION_PROMPT = """Given these two related documents:

Document 1 ({title1}):
{text1}

Document 2 ({title2}):
{text2}

Connection: {connection}

Generate 10 multi-hop questions that REQUIRE information from BOTH documents to answer correctly. Each question should not be answerable from just one document alone.

Output format (JSON array):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]

Make questions diverse: who/what/when/where/why/how. Output valid JSON only."""


def parse_json_array(text: str) -> list:
    """Extract JSON array from text."""
    try:
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return []


async def call_openrouter(client: AsyncOpenAI, prompt: str, max_tokens: int = 2000, temperature: float = 0.3) -> str:
    """Call OpenRouter API."""
    try:
        response = await client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"    API error: {e}")
        return ""


async def discover_links(client: AsyncOpenAI, passages: list[dict], semaphore: asyncio.Semaphore) -> list[dict]:
    """Use teacher model to discover linked passage pairs."""
    async with semaphore:
        docs_text = ""
        for i, p in enumerate(passages):
            docs_text += f"\n[Document {i}] {p['title']}:\n{p['text'][:500]}...\n"

        prompt = LINK_DISCOVERY_PROMPT.format(documents=docs_text)
        response = await call_openrouter(client, prompt, max_tokens=2000, temperature=0.3)

        links = parse_json_array(response)
        return links


async def generate_qa_for_pair(client: AsyncOpenAI, doc1: dict, doc2: dict, connection: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """Generate multi-hop QA pairs for a linked document pair."""
    async with semaphore:
        prompt = QA_GENERATION_PROMPT.format(
            title1=doc1['title'],
            text1=doc1['text'],
            title2=doc2['title'],
            text2=doc2['text'],
            connection=connection
        )

        response = await call_openrouter(client, prompt, max_tokens=3000, temperature=0.7)
        return parse_json_array(response)


async def create_type2_data(samples: list[dict], client: AsyncOpenAI) -> list[dict]:
    """Type 2: Discover links and generate multi-hop QA (excludes gold passages)."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    type2_data = []

    for idx, sample in enumerate(samples):
        print(f"  Processing sample {idx+1}/{len(samples)}: {sample['question'][:50]}...")

        passages = sample['ctxs']

        # Identify gold passage indices (to exclude)
        gold_titles = {p['title'] for p in sample.get('positive_ctxs', [])}
        gold_indices = {i for i, p in enumerate(passages) if p['title'] in gold_titles}

        # Discover links
        links = await discover_links(client, passages, semaphore)

        # Filter and generate QA
        qa_tasks = []
        valid_links = []
        for link in links:
            try:
                doc1_idx = link['doc1_idx']
                doc2_idx = link['doc2_idx']
                if doc1_idx >= len(passages) or doc2_idx >= len(passages):
                    continue
                if doc1_idx in gold_indices or doc2_idx in gold_indices:
                    continue
                qa_tasks.append(generate_qa_for_pair(
                    client, passages[doc1_idx], passages[doc2_idx],
                    link.get('connection', ''), semaphore
                ))
                valid_links.append(link.get('connection', ''))
            except (KeyError, IndexError, TypeError):
                continue

        if qa_tasks:
            qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)
            for qa_pairs in qa_results:
                if isinstance(qa_pairs, list):
                    for qa in qa_pairs:
                        if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                            type2_data.append({
                                "messages": [
                                    {"role": "user", "content": qa['question']},
                                    {"role": "assistant", "content": f"Answer: {qa['answer']}"}
                                ]
                            })

    return type2_data


def save_jsonl(data: list[dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} samples to {filepath}")


async def main_async():
    random.seed(42)

    print(f"Loading first {NUM_QUESTIONS} unique questions...")
    samples = load_first_n_unique_questions(INPUT_FILE, NUM_QUESTIONS)
    print(f"Loaded {len(samples)} samples")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Type 1: Passage absorption
    print("\n=== Creating Type 1 data (passage absorption) ===")
    type1_data = create_type1_data(samples)
    save_jsonl(type1_data, OUTPUT_DIR / "type1_passages.jsonl")

    # Type 2: Multi-hop QA generation
    print("\n=== Creating Type 2 data (multi-hop QA) ===")
    type2_data = await create_type2_data(samples, client)
    save_jsonl(type2_data, OUTPUT_DIR / "type2_multihop_qa.jsonl")

    # Combined (shuffled)
    combined = type1_data + type2_data
    random.shuffle(combined)
    save_jsonl(combined, OUTPUT_DIR / "combined_train.jsonl")

    print(f"\n=== Summary ===")
    print(f"Type 1 (passages): {len(type1_data)}")
    print(f"Type 2 (multi-hop QA): {len(type2_data)}")
    print(f"Combined: {len(combined)}")
    if len(type1_data) > 0:
        print(f"Ratio (Type2/Type1): {len(type2_data)/len(type1_data):.1f}x")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
