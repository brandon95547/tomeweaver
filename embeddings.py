"""
embeddings.py

Handles:
- SQLite storage of embeddings
- FAISS similarity index
- Hugging Face embedding calls
"""

import sqlite3
import time
from typing import Optional

import numpy as np
import requests
import faiss

from config import Config
from utils import serialize_vector, deserialize_vector, hash_text


class EmbeddingStore:
    def __init__(self, config: Config, db_path: str = "embeddings.db"):
        self.config = config
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._setup_table()
        self.index = self._load_faiss_index()
        self.headers = {"Authorization": f"Bearer {self.config.hf_api_key}"}

    # ---------- DB / FAISS setup ----------

    def _setup_table(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT,
                embedding BLOB
            )
        """)
        self.conn.commit()

    def _load_faiss_index(self) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(self.config.dimensions)
        self.cursor.execute("SELECT embedding FROM embeddings")
        rows = self.cursor.fetchall()

        vectors = []
        for (blob,) in rows:
            vec = deserialize_vector(blob)
            vectors.append(vec)

        if vectors:
            arr = np.array(vectors, dtype=np.float32)
            index.add(arr)

        return index

    # ---------- Embedding API ----------

    def get_embedding(
        self,
        text: str,
        retries: int = 3,
        delay: int = 5
    ) -> Optional[np.ndarray]:
        """
        Call Hugging Face feature-extraction API and return a float32 numpy array.
        Retries on failure. Returns None if all retries fail.
        """
        for attempt in range(retries):
            try:
                response = requests.post(
                    self.config.hf_endpoint,
                    headers=self.headers,
                    json={"inputs": text},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                # Handle [ [floats] ] or [floats]
                if isinstance(data, list) and data and isinstance(data[0], list):
                    vec = data[0]
                else:
                    vec = data

                return np.array(vec, dtype=np.float32)

            except Exception as e:
                print(f"[HuggingFace Error] Attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                else:
                    print("❌ Skipping due to repeated Hugging Face errors.")
                    return None

    # ---------- Similarity / duplicate checking ----------

    def is_duplicate(self, emb: np.ndarray) -> bool:
        """
        Return True if the given embedding is too similar to an existing one
        in FAISS (cosine/IP > similarity_threshold).
        """
        if self.index.ntotal == 0:
            return False

        emb = emb.reshape(1, -1).astype(np.float32)
        scores, _ = self.index.search(emb, k=1)
        return scores[0][0] > self.config.similarity_threshold

    # ---------- Insert new block ----------

    def add_block(self, block_text: str, emb: np.ndarray):
        """
        Insert a new text block + embedding into SQLite and FAISS.
        """
        block_hash = hash_text(block_text)
        emb_bytes = serialize_vector(emb)

        self.cursor.execute(
            "INSERT INTO embeddings (id, chunk, embedding) VALUES (?, ?, ?)",
            (block_hash, block_text, emb_bytes),
        )
        self.conn.commit()

        self.index.add(emb.reshape(1, -1).astype(np.float32))
