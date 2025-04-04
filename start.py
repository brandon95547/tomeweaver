# Import standard and third-party libraries
import os
import json
import requests
import hashlib
import sqlite3
import numpy as np
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import openai
import faiss
import re

# Load environment variables from a .env file
load_dotenv()

# Initialize DeepSeek client with API key and custom base URL
deepseek_client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Hugging Face API endpoint and authentication
HF_EMBEDDING_API = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# SQLite database and FAISS index setup
DB_PATH = "embeddings.db"
INDEX_PATH = "faiss.index"
DIMENSIONS = 384
SIMILARITY_THRESHOLD = 0.90

# Set up SQLite DB and create table for storing chunk embeddings
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    id TEXT PRIMARY KEY,
    chunk TEXT,
    embedding BLOB
)
""")
conn.commit()

# Utility: Hash a text string (for deduplication)
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Utility: Serialize a list of floats into bytes (for SQLite storage)
def serialize_vector(vec: List[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()

# Utility: Deserialize bytes back into a NumPy array
def deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

# Get sentence embedding using Hugging Face API with retries
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
                raise ValueError("Invalid embedding format")
        except Exception as e:
            print(f"[HuggingFace Error] Attempt {attempt + 1} of {retries}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print("\u274c Skipping due to repeated Hugging Face errors.")
                return None

# Load all stored embeddings from SQLite into FAISS index for similarity search
def load_faiss_index() -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(DIMENSIONS)
    vectors = []
    cursor.execute("SELECT embedding FROM embeddings")
    for (blob,) in cursor.fetchall():
        vec = deserialize_vector(blob)
        vectors.append(vec)
    if vectors:
        index.add(np.array(vectors, dtype=np.float32))
    return index

faiss_index = load_faiss_index()

# Load input text to be organized
input_file = Path("tmp/pizza.txt")
with input_file.open("r", encoding="utf-8") as f:
    full_text = f.read()

# Split large input text into smaller chunks
def split_text_into_chunks(text: str, max_chars: int = 10000) -> List[str]:
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

# Load Table of Contents
full_md_path = Path("toc/full.md")
with full_md_path.open("r", encoding="utf-8") as f:
    full_md_lines = f.readlines()

# Extract headings
toc_headings_list = [line.strip() for line in full_md_lines if line.strip().startswith("#")]
toc_heading_set = set(line.lstrip("#").strip() for line in toc_headings_list)

prompt_template = (
    "You are a professional book editor working with the following existing Table of Contents:\n"
    + "\n".join(toc_headings_list) +
    "\n\nYour task is to read the provided raw text and do the following:\n"
    "1. Identify ALL headings and subheadings from the Table of Contents that are relevant to this chunk.\n"
    "2. For each relevant heading, organize the chunk’s content underneath it using logical structure (PART > Chapter > Subsection > Bullet points).\n"
    "3. Repeat this for each matching heading, if applicable.\n"
    "4. Do NOT invent new top-level sections or change the original ToC.\n"
    "5. Use markdown formatting consistently.\n\n"
    "Output format:\n"
    "# Matching TOC Heading A\n"
    "(structured info here...)\n\n"
    "# Matching TOC Heading B\n"
    "(more structured info...)\n\n"
    "Text: {chunk}"
)

organized_sections = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")

    prompt = prompt_template.format(chunk=chunk)
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    organized_output = response.choices[0].message.content
    organized_sections.append(organized_output)

# Insert all structured sections
heading_pattern = re.compile(r"^(#+)\s+(.*)")

for section in organized_sections:
    lines = section.strip().splitlines()
    i = 0
    while i < len(lines):
        match = heading_pattern.match(lines[i])
        if match:
            heading_level, heading_text = match.groups()
            normalized_heading = heading_text.strip()
            content_lines = []

            i += 1
            while i < len(lines) and not heading_pattern.match(lines[i]):
                content_lines.append(lines[i])
                i += 1

            if normalized_heading not in toc_heading_set:
                print(f"Warning: Heading '{normalized_heading}' not found in toc/full.md. Skipping.")
                continue

            new_block = "\n" + "\n".join(content_lines).strip() + "\n"
            print(f"new_block -- {new_block}")

            # Similarity check using new_block
            new_block_embedding = get_embedding(new_block)
            if new_block_embedding is None:
                print("-> Skipping new_block due to embedding error.")
                continue

            new_block_emb = np.array(new_block_embedding, dtype=np.float32).reshape(1, -1)

            if faiss_index.ntotal > 0:
                scores, _ = faiss_index.search(new_block_emb, k=1)
                if scores[0][0] > SIMILARITY_THRESHOLD:
                    print(f"-> Skipping new_block under '{normalized_heading}' (FAISS duplicate)")
                    continue

            # Insert block and update FAISS + DB
            try:
                heading_index = next(
                    idx for idx, line in enumerate(full_md_lines)
                    if line.lstrip("#").strip() == normalized_heading
                )
                full_md_lines.insert(heading_index + 1, new_block)

                block_hash = hash_text(new_block)
                cursor.execute("INSERT INTO embeddings (id, chunk, embedding) VALUES (?, ?, ?)", (
                    block_hash,
                    new_block,
                    serialize_vector(new_block_emb.flatten())
                ))
                conn.commit()
                faiss_index.add(new_block_emb)

            except StopIteration:
                print(f"Warning: Heading '{normalized_heading}' not found in toc/full.md. Skipping.")
        else:
            i += 1

with full_md_path.open("w", encoding="utf-8") as f:
    f.writelines(full_md_lines)

print("\u2705 Updated toc/full.md and embeddings database.")
