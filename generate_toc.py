import os
import openai
import json
import requests
from pathlib import Path
from typing import List, Set
from dotenv import load_dotenv
import re
import time

# Load environment variables from .env
load_dotenv()

deepseek_client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

HF_EMBEDDING_API = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

input_file = Path("tmp/pizza.txt")
with input_file.open("r", encoding="utf-8") as f:
    full_text = f.read()

def split_text_into_chunks(text: str, max_chars: int = 8000) -> List[str]:
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind('.', 0, max_chars)
        if split_at == -1:
            split_at = text.rfind('\n', 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at + 1].strip())
        text = text[split_at + 1:].strip()
    if text:
        chunks.append(text)
    return chunks

chunks = split_text_into_chunks(full_text)
print(f"Split into {len(chunks)} chunks.")

SIMILARITY_THRESHOLD = 0.90
seen_embeddings = []

def get_embedding(text: str, retries: int = 3, delay: int = 5) -> List[float]:
    for attempt in range(retries):
        try:
            response = requests.post(HF_EMBEDDING_API, headers=HEADERS, json={"inputs": text}, timeout=30)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and isinstance(data[0], list):
                return data[0]
            elif isinstance(data, list):
                return data
            else:
                raise ValueError("Invalid embedding format returned from Hugging Face API")
        except Exception as e:
            print(f"[HuggingFace Error] Attempt {attempt + 1} of {retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("\u274c Skipping embedding due to repeated Hugging Face errors.")
                return None

def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2)

toc_prompt_template = (
    "From the following text, extract a clean and descriptive Table of Contents.\n\n"
    "Guidelines:\n"
    "- Headings should accurately reflect what the chunk is about.\n"
    "- Do not use generic titles like “Chapter 3” or “Part II” unless they appear in the text.\n"
    "- Create specific, content-driven headings.\n"
    "- Keep the output to only headings and subheadings (topics and subtopics).\n"
    "- Format using only '#' for main headings and '##' for subheadings.\n\n"
    "Text:\n{chunk}"
)

toc_sections: List[str] = []
heading_texts: Set[str] = set()

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}/{len(chunks)}...")
    prompt = toc_prompt_template.format(chunk=chunk)
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    toc_output = response.choices[0].message.content
    for line in toc_output.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("#"):
            continue

        embedding = get_embedding(stripped)
        if embedding is None:
            continue

        if any(cosine_similarity(embedding, seen) > SIMILARITY_THRESHOLD for seen in seen_embeddings):
            continue

        if stripped not in heading_texts:
            heading_texts.add(stripped)
            seen_embeddings.append(embedding)
            toc_sections.append(stripped)

final_prompt = (
    "You are a professional book editor.\n"
    "Organize the following list of headings into a complete and coherent Table of Contents for a structured investigation.\n"
    "Group similar topics under appropriate Parts, with a clear hierarchy using only '#', '##', and '###'.\n"
    "Do not change the wording of the headings.\n"
    "Preserve the order as much as possible, but feel free to group logically.\n"
    "Here are the extracted headings:\n\n"
    f"{chr(10).join(toc_sections)}"
)

response = deepseek_client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.3
)

grouped_toc = response.choices[0].message.content

# Save full version to a file
Path("toc").mkdir(parents=True, exist_ok=True)
with Path("toc/full.md").open("w", encoding="utf-8") as f:
    f.write(grouped_toc)

# Save individual top-level sections to files
current_title = None
section_lines = []

for line in grouped_toc.splitlines():
    if line.startswith("# "):
        if current_title:
            filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", current_title.strip("# ")).lower()
            with Path(f"toc/{filename}.md").open("w", encoding="utf-8") as f:
                f.write("\n".join(section_lines))
        current_title = line
        section_lines = [line]
    elif current_title:
        section_lines.append(line)

# Save last section
if current_title:
    filename = re.sub(r"[^a-zA-Z0-9_-]+", "_", current_title.strip("# ")).lower()
    with Path(f"toc/{filename}.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(section_lines))

print("Individual TOC files saved in ./toc directory")
