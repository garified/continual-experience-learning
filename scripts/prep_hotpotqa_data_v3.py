"""
Prepare HotpotQA training data v3: Multi-variant passage synthesis.

Type 1: 15 paraphrased variants per passage using explicit styles
Type 2: Multi-hop QA generation (same as v2)

Uses OpenRouter for teacher model with parallel inference.
"""

import json
import asyncio
import os
import random
from pathlib import Path
from openai import AsyncOpenAI

INPUT_FILE = "/sfs/weka/scratch/ks8vf/exp/HELMET/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl"
OUTPUT_DIR = Path("/sfs/weka/scratch/ks8vf/exp/data/hotpotqa_v3")
TEACHER_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
NUM_QUESTIONS = 20
MAX_CONCURRENT = 10
NUM_VARIANTS = 15
RANDOM_SEED = 42

# 15 explicit styles for paraphrasing
STYLES = [
    ("formal_academic", "Rewrite this passage in formal academic prose, suitable for a scholarly article. Use precise language and maintain objectivity."),
    ("casual_blog", "Rewrite this passage as a casual blog post. Use conversational tone, contractions, and engaging language."),
    ("bullet_points", "Rewrite this passage as a bullet-point summary. Extract key facts as concise bullet points."),
    ("qa_format", "Rewrite this passage as a Q&A format. Create questions about the key facts and answer them."),
    ("narrative_story", "Rewrite this passage as a narrative story. Add flow and storytelling elements while preserving all facts."),
    ("technical_doc", "Rewrite this passage as technical documentation. Use clear, structured language with precise definitions."),
    ("news_article", "Rewrite this passage as a news article. Use journalistic style with the most important information first."),
    ("wikipedia_style", "Rewrite this passage in Wikipedia style. Use encyclopedic tone with neutral point of view."),
    ("textbook", "Rewrite this passage as a textbook explanation. Make it educational and easy to understand."),
    ("dialogue", "Rewrite this passage as a dialogue between two people discussing the topic. Preserve all key facts in the conversation."),
    ("executive_summary", "Rewrite this passage as a brief executive summary. Be extremely concise while keeping essential facts."),
    ("detailed_elaboration", "Rewrite this passage with more detail and elaboration. Expand on the facts while maintaining accuracy."),
    ("first_person", "Rewrite this passage from a first-person perspective, as if narrating personal knowledge of the topic."),
    ("timeline", "Rewrite this passage emphasizing chronological order. Organize facts by time sequence where applicable."),
    ("compare_contrast", "Rewrite this passage using compare/contrast framing. Highlight relationships and distinctions between elements."),
]

PARAPHRASE_PROMPT = """Rewrite the following passage about "{title}".

IMPORTANT: Preserve ALL key facts, names, dates, and specific information. The meaning must be identical, only the style changes.

Style instruction: {style_instruction}

Original passage:
{text}

Rewritten passage:"""


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


async def call_openrouter(client: AsyncOpenAI, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
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


async def generate_variant(client: AsyncOpenAI, title: str, text: str, style_name: str, style_instruction: str, semaphore: asyncio.Semaphore) -> dict | None:
    """Generate one paraphrased variant of a passage."""
    async with semaphore:
        prompt = PARAPHRASE_PROMPT.format(
            title=title,
            style_instruction=style_instruction,
            text=text
        )

        response = await call_openrouter(client, prompt, max_tokens=1000, temperature=0.7)

        if not response.strip():
            return None

        return {
            "style": style_name,
            "text": response.strip()
        }


async def generate_all_variants(client: AsyncOpenAI, title: str, text: str, semaphore: asyncio.Semaphore) -> list[dict]:
    """Generate all 15 variants for a passage."""
    tasks = [
        generate_variant(client, title, text, style_name, style_instruction, semaphore)
        for style_name, style_instruction in STYLES
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    variants = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"      Error generating {STYLES[i][0]}: {result}")
        elif result is not None:
            variants.append(result)

    return variants


async def create_type1_variants(samples: list[dict], client: AsyncOpenAI) -> tuple[list[dict], dict]:
    """
    Type 1 v3: Generate 15 paraphrased variants for each unique passage.
    Returns (all_variants_data, passage_to_variants_map)
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Collect unique passages
    seen_passages = {}
    for sample in samples:
        for ctx in sample['ctxs']:
            psg_id = ctx.get('id', ctx['title'])
            if psg_id not in seen_passages:
                seen_passages[psg_id] = {
                    'title': ctx['title'],
                    'text': ctx['text'],
                    'id': psg_id
                }

    print(f"  Found {len(seen_passages)} unique passages")

    # Generate variants for each passage
    passage_variants = {}
    all_variants = []

    for i, (psg_id, psg) in enumerate(seen_passages.items()):
        print(f"  Processing passage {i+1}/{len(seen_passages)}: {psg['title'][:40]}...")

        variants = await generate_all_variants(client, psg['title'], psg['text'], semaphore)

        # Randomize order of variants (seeded for reproducibility)
        random.seed(RANDOM_SEED + hash(psg_id))
        random.shuffle(variants)

        passage_variants[psg_id] = variants

        # Create training samples for each variant
        for j, variant in enumerate(variants):
            all_variants.append({
                "messages": [
                    {"role": "user", "content": f"Tell me about {psg['title']}"},
                    {"role": "assistant", "content": variant['text']}
                ],
                "metadata": {
                    "passage_id": psg_id,
                    "variant_idx": j,
                    "style": variant['style'],
                    "original_title": psg['title']
                }
            })

        print(f"    Generated {len(variants)} variants")

    return all_variants, passage_variants


def create_type1_subset(all_variants: list[dict], num_variants_per_passage: int) -> list[dict]:
    """Create a subset using first N variants per passage (already randomized)."""
    subset = []
    passage_counts = {}

    for item in all_variants:
        psg_id = item['metadata']['passage_id']
        variant_idx = item['metadata']['variant_idx']

        if variant_idx < num_variants_per_passage:
            subset.append(item)
            passage_counts[psg_id] = passage_counts.get(psg_id, 0) + 1

    return subset


# Type 2 code (same as v2)
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
    """Type 2: Discover links and generate multi-hop QA (same as v2)."""
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
    random.seed(RANDOM_SEED)

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

    # Type 1: Multi-variant passages
    print("\n=== Creating Type 1 data (15 variants per passage) ===")
    all_type1_variants, passage_variants = await create_type1_variants(samples, client)
    save_jsonl(all_type1_variants, OUTPUT_DIR / "type1_all_variants.jsonl")

    # Save variant subsets (5, 10, 15)
    for n_variants in [5, 10, 15]:
        subset = create_type1_subset(all_type1_variants, n_variants)
        save_jsonl(subset, OUTPUT_DIR / f"type1_{n_variants}variants.jsonl")

    # Type 2: Multi-hop QA (same as before)
    print("\n=== Creating Type 2 data (multi-hop QA) ===")
    type2_data = await create_type2_data(samples, client)
    save_jsonl(type2_data, OUTPUT_DIR / "type2_multihop_qa.jsonl")

    # Combined datasets for each variant count
    print("\n=== Creating combined datasets ===")
    for n_variants in [5, 10, 15]:
        type1_subset = create_type1_subset(all_type1_variants, n_variants)
        combined = type1_subset + type2_data
        random.seed(RANDOM_SEED)
        random.shuffle(combined)
        save_jsonl(combined, OUTPUT_DIR / f"combined_{n_variants}var.jsonl")

    print(f"\n=== Summary ===")
    print(f"Type 1 (all 15 variants): {len(all_type1_variants)}")
    print(f"Type 1 (5 variants): {len(create_type1_subset(all_type1_variants, 5))}")
    print(f"Type 1 (10 variants): {len(create_type1_subset(all_type1_variants, 10))}")
    print(f"Type 2 (multi-hop QA): {len(type2_data)}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
