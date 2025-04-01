import os
import json
import requests
import hashlib
import sqlite3
import numpy as np
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import openai
import faiss

# Load .env
load_dotenv()

# OpenAI DeepSeek
deepseek_client = openai.OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Hugging Face Embedding API
HF_EMBEDDING_API = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# SQLite + FAISS setup
DB_PATH = "embeddings.db"
INDEX_PATH = "faiss.index"
DIMENSIONS = 384  # MiniLM-L6-v2 returns 384-dim vectors
SIMILARITY_THRESHOLD = 0.90

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

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def serialize_vector(vec: List[float]) -> bytes:
    return np.array(vec, dtype=np.float32).tobytes()

def deserialize_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)

def get_embedding(text: str) -> List[float]:
    response = requests.post(HF_EMBEDDING_API, headers=HEADERS, json={"inputs": text})
    data = response.json()
    if isinstance(data, list) and isinstance(data[0], list):
        return data[0]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Invalid embedding format")

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return float(dot / (norm1 * norm2))

# Load existing embeddings into FAISS
def load_faiss_index() -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(DIMENSIONS)
    ids = []
    vectors = []
    cursor.execute("SELECT id, embedding FROM embeddings")
    for row_id, blob in cursor.fetchall():
        vec = deserialize_vector(blob)
        ids.append(row_id)
        vectors.append(vec)
    if vectors:
        index.add(np.array(vectors, dtype=np.float32))
    return index

faiss_index = load_faiss_index()

# Load text file
input_file = Path("tmp/pizza.txt")
with input_file.open("r", encoding="utf-8") as f:
    full_text = f.read()

def split_text_into_chunks(text: str, max_chars: int = 5000) -> List[str]:
    chunks = []
    while len(text) > max_chars:
        split_at = text.rfind("\n", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks

chunks = split_text_into_chunks(full_text, max_chars=5000)
print(f"Split into {len(chunks)} chunks.")

# Load TOC and normalize headings
full_md_path = Path("toc/full.md")
with full_md_path.open("r", encoding="utf-8") as f:
    full_md_lines = f.readlines()

toc_headings_list = [line.strip() for line in full_md_lines if line.strip().startswith("#")]
toc_heading_set = set(line.lstrip("#").strip() for line in toc_headings_list)

# Prompt template
prompt_template = (
    "You are a professional book editor working with the following existing Table of Contents:\n"
    + "\n".join(toc_headings_list) +
    "\nYour task is to read the provided raw text and do the following:\n"
    "1. Choose the most relevant heading or subheading from the Table of Contents.\n"
    "2. Place that heading at the top of your response.\n"
    "3. Organize the chunk’s content underneath and expand as much as possible using logical structure (PART > Chapter > Subsection > Bullet points).\n"
    "4. Do not create new top-level sections ever.\n"
    "5. Keep formatting markdown-compliant.\n\n"
    "Text: {chunk}"
)

organized_sections = []

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")

    chunk_hash = hash_text(chunk)
    cursor.execute("SELECT 1 FROM embeddings WHERE id = ?", (chunk_hash,))
    if cursor.fetchone():
        print("-> Skipping chunk (already in DB)")
        continue

    chunk_emb = np.array(get_embedding(chunk), dtype=np.float32).reshape(1, -1)
    if faiss_index.ntotal > 0:
        scores, _ = faiss_index.search(chunk_emb, k=1)
        if scores[0][0] > SIMILARITY_THRESHOLD:
            print("-> Skipping chunk (FAISS match found)")
            continue

    # Prepare AI prompt
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

    # Save to DB + FAISS
    cursor.execute("INSERT INTO embeddings (id, chunk, embedding) VALUES (?, ?, ?)", (
        chunk_hash,
        chunk,
        serialize_vector(chunk_emb.flatten())
    ))
    conn.commit()
    faiss_index.add(chunk_emb)

# Insert organized output into full.md
for section in organized_sections:
    lines = section.strip().splitlines()
    if not lines:
        continue
    heading_line = lines[0].strip()
    content_lines = lines[1:]
    normalized_heading = heading_line.lstrip("#").strip()

    if normalized_heading not in toc_heading_set:
        print(f"Warning: Heading '{heading_line}' not found in toc/full.md. Skipping.")
        continue

    try:
        insert_index = next(
            i for i, line in enumerate(full_md_lines)
            if line.lstrip("#").strip() == normalized_heading
        )
        insert_index += 1
        while insert_index < len(full_md_lines) and not (
            full_md_lines[insert_index].strip().startswith("#")
            or full_md_lines[insert_index].strip() == ""
        ):
            insert_index += 1
        full_md_lines.insert(insert_index, "\n" + "\n".join(content_lines) + "\n")
    except StopIteration:
        print(f"Warning: Heading '{heading_line}' not found in toc/full.md. Skipping.")

# Save updated Markdown
with full_md_path.open("w", encoding="utf-8") as f:
    f.writelines(full_md_lines)

print("✅ Updated toc/full.md and embeddings database.")
