"""
utils.py

General utility functions used across the project:
- Hashing text
- Serializing / deserializing vectors
- Loading text files
- Splitting large text into reasonably sized chunks
"""

import hashlib
from pathlib import Path
from typing import List, Union

import numpy as np


# ---------- Hashing ----------

def hash_text(text: str) -> str:
    """
    Return a stable SHA-256 hex digest for the given text.
    Used as a unique ID for content blocks.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- Embedding (vector) serialization ----------

def serialize_vector(vec: Union[List[float], np.ndarray]) -> bytes:
    """
    Serialize a vector (list or NumPy array) into bytes suitable for SQLite storage.

    - Ensures dtype float32
    - Returns raw bytes via .tobytes()
    """
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """
    Deserialize bytes from SQLite back into a float32 NumPy array.
    """
    return np.frombuffer(blob, dtype=np.float32)


# ---------- File loading ----------

def load_text(path: Union[str, Path]) -> str:
    """
    Load a UTF-8 text file and return its full contents as a string.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return f.read()


# ---------- Text chunking ----------

def split_text_into_chunks(text: str, max_chars: int = 10000) -> List[str]:
    """
    Split a large text into smaller chunks no longer than `max_chars`,
    preferably breaking at sentence or line boundaries.

    Strategy:
    - While text is longer than max_chars:
        - Try to split at the last '.' before max_chars
        - If no '.', try the last newline before max_chars
        - If neither is found, hard-split at max_chars
    - Strip leading/trailing whitespace from each chunk.
    """
    chunks: List[str] = []
    text = text.strip()

    while len(text) > max_chars:
        # Prefer splitting at a period before max_chars
        split_at = text.rfind('.', 0, max_chars)
        if split_at == -1:
            # Fallback: split at newline
            split_at = text.rfind('\n', 0, max_chars)
        if split_at == -1:
            # As a last resort, hard-split at max_chars
            split_at = max_chars

        # Include the delimiter (e.g., '.') in the chunk
        chunk = text[: split_at + 1].strip()
        if chunk:
            chunks.append(chunk)

        # Remaining text
        text = text[split_at + 1 :].strip()

    # Any leftover text becomes the final chunk
    if text:
        chunks.append(text)

    return chunks
