"""
Prepare HotpotQA training data for knowledge extraction SFT.

Type 1: Passage absorption (chat pairs from ctxs)
Type 2: Multi-hop QA generation (teacher discovers links, generates questions)
"""

import json
import asyncio
from pathlib import Path

# Tinker imports for teacher model
from tinker import SamplingClient, ServiceClient

INPUT_FILE = "/sfs/weka/scratch/ks8vf/exp/HELMET/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl"
OUTPUT_DIR = Path("/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_20")
TEACHER_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
NUM_QUESTIONS = 20


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
            # Dedupe by passage id
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


async def discover_links(sampling_client: SamplingClient, passages: list[dict]) -> list[dict]:
    """Use teacher model to discover linked passage pairs."""
    # Format documents
    docs_text = ""
    for i, p in enumerate(passages):
        docs_text += f"\n[Document {i}] {p['title']}:\n{p['text'][:500]}...\n"

    prompt = LINK_DISCOVERY_PROMPT.format(documents=docs_text)

    response = await sampling_client.completions(
        prompt=prompt,
        max_tokens=2000,
        temperature=0.3,
    )

    # Parse JSON from response
    try:
        # Find JSON array in response
        text = response.completion
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        print(f"Failed to parse links: {response.completion[:200]}")
    return []


async def generate_qa_for_pair(
    sampling_client: SamplingClient,
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

    response = await sampling_client.completions(
        prompt=prompt,
        max_tokens=3000,
        temperature=0.7,
    )

    # Parse JSON from response
    try:
        text = response.completion
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        print(f"Failed to parse QA: {response.completion[:200]}")
    return []


async def create_type2_data(samples: list[dict]) -> list[dict]:
    """
    Type 2: Discover links and generate multi-hop QA.
    """
    service_client = ServiceClient(model_name=TEACHER_MODEL)
    sampling_client = service_client.make_sampling_client()

    type2_data = []

    for i, sample in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}: {sample['question'][:50]}...")

        passages = sample['ctxs']

        # Step 1: Discover linked pairs
        links = await discover_links(sampling_client, passages)
        print(f"  Found {len(links)} linked pairs")

        # Step 2: Generate QA for each linked pair
        for link in links:
            try:
                doc1 = passages[link['doc1_idx']]
                doc2 = passages[link['doc2_idx']]
                connection = link.get('connection', '')

                qa_pairs = await generate_qa_for_pair(
                    sampling_client, doc1, doc2, connection
                )

                for qa in qa_pairs:
                    type2_data.append({
                        "messages": [
                            {"role": "user", "content": qa['question']},
                            {"role": "assistant", "content": f"Answer: {qa['answer']}"}
                        ]
                    })
                print(f"    Generated {len(qa_pairs)} QA pairs for link: {connection[:50]}")
            except (KeyError, IndexError) as e:
                print(f"    Skipping invalid link: {e}")

    return type2_data


def save_jsonl(data: list[dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(data)} samples to {filepath}")


async def main():
    print(f"Loading first {NUM_QUESTIONS} unique questions...")
    samples = load_first_n_unique_questions(INPUT_FILE, NUM_QUESTIONS)
    print(f"Loaded {len(samples)} samples")

    # Type 1: Passage absorption
    print("\n=== Creating Type 1 data (passage absorption) ===")
    type1_data = create_type1_data(samples)
    save_jsonl(type1_data, OUTPUT_DIR / "type1_passages.jsonl")

    # Type 2: Multi-hop QA generation
    print("\n=== Creating Type 2 data (multi-hop QA) ===")
    type2_data = await create_type2_data(samples)
    save_jsonl(type2_data, OUTPUT_DIR / "type2_multihop_qa.jsonl")

    # Combined
    combined = type1_data + type2_data
    save_jsonl(combined, OUTPUT_DIR / "combined_train.jsonl")

    print(f"\n=== Summary ===")
    print(f"Type 1 (passages): {len(type1_data)}")
    print(f"Type 2 (multi-hop QA): {len(type2_data)}")
    print(f"Combined: {len(combined)}")
    print(f"Ratio (Type2/Type1): {len(type2_data)/len(type1_data):.1f}x")


if __name__ == "__main__":
    asyncio.run(main())
