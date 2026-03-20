"""
Prepare HotpotQA training data for v5: per-slice knowledge extraction.

5 non-overlapping slices of 20 questions each from the 100 HELMET questions.
Slice 1 (Q1-20) can reuse existing data/hotpotqa_20/ (identical to v2).

Usage:
    python scripts/prep_hotpotqa_data_v5.py --slice 2
    python scripts/prep_hotpotqa_data_v5.py --slice 3
"""

import json
import asyncio
import argparse
import os
import random
from pathlib import Path
from openai import AsyncOpenAI

INPUT_FILE = "/sfs/weka/scratch/ks8vf/exp/HELMET/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl"
TEACHER_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
NUM_QUESTIONS_PER_SLICE = 20
MAX_CONCURRENT = 10


def load_slice_questions(filepath: str, slice_num: int) -> list[dict]:
    """Load 20 unique questions for a given slice (1-5).

    Slice 1: questions 1-20, Slice 2: questions 21-40, etc.
    Returns one sample per unique question (first depth only), matching v2 behavior.
    """
    with open(filepath) as f:
        data = [json.loads(line) for line in f]

    # Collect all unique questions in order (first occurrence = first depth)
    seen = set()
    all_unique = []
    for d in data:
        if d['question'] not in seen:
            seen.add(d['question'])
            all_unique.append(d)

    skip = (slice_num - 1) * NUM_QUESTIONS_PER_SLICE
    end = skip + NUM_QUESTIONS_PER_SLICE
    if end > len(all_unique):
        raise ValueError(f"Not enough unique questions: need {end}, have {len(all_unique)}")

    return all_unique[skip:end]


def create_type1_data(samples: list[dict]) -> list[dict]:
    """Type 1: Convert passages to chat format for knowledge absorption."""
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


async def discover_links(client: AsyncOpenAI, passages: list[dict]) -> list[dict]:
    """Use teacher model to discover linked passage pairs."""
    docs_text = ""
    for i, p in enumerate(passages):
        docs_text += f"\n[Document {i}] {p['title']}:\n{p['text'][:500]}...\n"

    prompt = LINK_DISCOVERY_PROMPT.format(documents=docs_text)
    response = await call_openrouter(client, prompt, max_tokens=2000, temperature=0.3)

    links = parse_json_array(response)
    if not links:
        print(f"    Failed to parse links from: {response[:200]}")
    return links


async def generate_qa_for_pair(
    client: AsyncOpenAI,
    doc1: dict,
    doc2: dict,
    connection: str
) -> list[dict]:
    """Generate multi-hop QA pairs for a linked document pair."""
    prompt = QA_GENERATION_PROMPT.format(
        title1=doc1['title'],
        text1=doc1['text'],
        title2=doc2['title'],
        text2=doc2['text'],
        connection=connection
    )

    response = await call_openrouter(client, prompt, max_tokens=3000, temperature=0.7)

    qa_pairs = parse_json_array(response)
    if not qa_pairs:
        print(f"    Failed to parse QA from: {response[:200]}")
    return qa_pairs


async def process_sample(client: AsyncOpenAI, sample: dict, idx: int, total: int, semaphore: asyncio.Semaphore) -> list[dict]:
    """Process a single sample with rate limiting."""
    async with semaphore:
        print(f"Processing sample {idx+1}/{total}: {sample['question'][:50]}...")

        passages = sample['ctxs']
        sample_qa_data = []

        gold_titles = {p['title'] for p in sample.get('positive_ctxs', [])}
        gold_indices = {i for i, p in enumerate(passages) if p['title'] in gold_titles}
        print(f"  Gold passages at indices: {gold_indices}")

        links = await discover_links(client, passages)
        print(f"  Found {len(links)} linked pairs (before filtering)")

        qa_tasks = []
        valid_links = []
        filtered_count = 0
        for link in links:
            try:
                doc1_idx = link['doc1_idx']
                doc2_idx = link['doc2_idx']
                if doc1_idx >= len(passages) or doc2_idx >= len(passages):
                    continue
                if doc1_idx in gold_indices or doc2_idx in gold_indices:
                    filtered_count += 1
                    continue
                doc1 = passages[doc1_idx]
                doc2 = passages[doc2_idx]
                connection = link.get('connection', '')
                qa_tasks.append(generate_qa_for_pair(client, doc1, doc2, connection))
                valid_links.append(connection)
            except (KeyError, IndexError, TypeError):
                continue
        print(f"  Filtered out {filtered_count} links involving gold passages, {len(qa_tasks)} remaining")

        if qa_tasks:
            qa_results = await asyncio.gather(*qa_tasks, return_exceptions=True)

            for connection, qa_pairs in zip(valid_links, qa_results):
                if isinstance(qa_pairs, Exception):
                    print(f"    Error for link: {qa_pairs}")
                    continue
                for qa in qa_pairs:
                    if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                        sample_qa_data.append({
                            "messages": [
                                {"role": "user", "content": qa['question']},
                                {"role": "assistant", "content": f"Answer: {qa['answer']}"}
                            ]
                        })
                print(f"    Generated {len(qa_pairs) if isinstance(qa_pairs, list) else 0} QA pairs for link: {connection[:50]}")

        return sample_qa_data


async def create_type2_data(samples: list[dict], client: AsyncOpenAI) -> list[dict]:
    """Type 2: Discover links and generate multi-hop QA with parallel processing."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        process_sample(client, sample, i, len(samples), semaphore)
        for i, sample in enumerate(samples)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    type2_data = []
    for result in results:
        if isinstance(result, Exception):
            print(f"Sample error: {result}")
        else:
            type2_data.extend(result)

    return type2_data


def save_jsonl(data: list[dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} samples to {filepath}")


async def main_async():
    parser = argparse.ArgumentParser(description="Prepare v5 slice data")
    parser.add_argument("--slice", type=int, required=True, choices=[1, 2, 3, 4, 5],
                        help="Slice number (1-5)")
    args = parser.parse_args()

    slice_num = args.slice
    output_dir = Path(f"/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_v5_s{slice_num}")

    # Slice 1 reuses existing v2 data
    if slice_num == 1:
        existing = Path("/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_20/combined_train.jsonl")
        if existing.exists():
            print(f"Slice 1 reuses existing data at {existing}")
            print(f"To use it for training, point to: {existing}")
            print(f"Or create a symlink: ln -s {existing.parent} {output_dir}")
            return
        else:
            print("Existing slice 1 data not found, generating fresh...")

    print(f"Loading slice {slice_num} questions (Q{(slice_num-1)*20+1}-{slice_num*20})...")
    samples = load_slice_questions(INPUT_FILE, slice_num)
    unique_questions = len(set(s['question'] for s in samples))
    print(f"Loaded {len(samples)} samples ({unique_questions} unique questions)")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    # Type 1: Passage absorption
    print(f"\n=== Creating Type 1 data (passage absorption) ===")
    type1_data = create_type1_data(samples)
    save_jsonl(type1_data, output_dir / "type1_passages.jsonl")

    # Type 2: Multi-hop QA generation
    print(f"\n=== Creating Type 2 data (multi-hop QA) ===")
    print(f"Using {MAX_CONCURRENT} concurrent API calls")
    type2_data = await create_type2_data(samples, client)
    save_jsonl(type2_data, output_dir / "type2_multihop_qa.jsonl")

    # Combined (shuffled)
    combined = type1_data + type2_data
    random.seed(42)
    random.shuffle(combined)
    save_jsonl(combined, output_dir / "combined_train.jsonl")

    print(f"\n=== Summary (slice {slice_num}) ===")
    print(f"Type 1 (passages): {len(type1_data)}")
    print(f"Type 2 (multi-hop QA): {len(type2_data)}")
    print(f"Combined: {len(combined)}")
    if len(type1_data) > 0:
        print(f"Ratio (Type2/Type1): {len(type2_data)/len(type1_data):.1f}x")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
