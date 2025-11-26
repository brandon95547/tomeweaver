# toc_generator.py

"""
toc_generator.py

Builds a candidate Table of Contents (TOC) from raw book text using DeepSeek + local
embeddings, and writes ONLY:
- toc/full.md

No per-heading markdown files are created in this version.
"""

from pathlib import Path
from typing import List, Set

import numpy as np
import re

from .utils import split_text_into_chunks
from .embeddings import EmbeddingStore


class TocGenerator:
    def __init__(self, config, embedding_store: EmbeddingStore):
        self.config = config
        self.embedding_store = embedding_store
        self.client = config.deepseek_client
        self.similarity_threshold = float(config.similarity_threshold)

    # ------------------------------------------------------------------
    # Step 1: Extract headings (DeepSeek + embeddings dedupe)
    # ------------------------------------------------------------------

    def extract_headings(
        self,
        full_text: str,
        max_chars_per_chunk: int = 10_000,
    ) -> List[str]:
        """
        Use DeepSeek to extract candidate headings from each text chunk,
        dedupe using cosine similarity, return a unique flat list.
        """

        chunks = split_text_into_chunks(full_text, max_chars=max_chars_per_chunk)
        print(f"[TOC] Split text into {len(chunks)} chunks for heading extraction.")

        seen_embeddings: List[np.ndarray] = []
        heading_texts: Set[str] = set()
        toc_sections: List[str] = []

        toc_prompt_template = (
            "You are extracting candidate section headings from a long non-fiction text.\n"
            "Given the following chunk, propose a short list of Markdown headings that "
            "describe the main topics.\n\n"
            "Rules:\n"
            "- Return ONLY Markdown headings, one per line.\n"
            "- Use only '#', '##', or '###' at the start of each heading.\n"
            "- No commentary or explanation, just headings.\n\n"
            "Chunk:\n"
            "{chunk}"
        )

        for idx, chunk in enumerate(chunks, start=1):
            print(f"[TOC] Processing chunk {idx}/{len(chunks)} for headings...")

            prompt = toc_prompt_template.format(chunk=chunk)

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            result = response.choices[0].message.content
            if not result:
                continue

            for line in result.splitlines():
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue

                # Normalize heading
                heading_text = stripped.lstrip("#").strip()
                if not heading_text:
                    continue

                # Embed heading
                emb = self.embedding_store.get_embedding(heading_text)
                if emb is None:
                    continue

                # Check duplicates via cosine similarity
                if any(
                    self._cosine_similarity(emb, prev) >= self.similarity_threshold
                    for prev in seen_embeddings
                ):
                    continue

                # Accept heading
                if heading_text not in heading_texts:
                    heading_texts.add(heading_text)
                    seen_embeddings.append(emb)
                    toc_sections.append(heading_text)

        print(f"[TOC] Collected {len(toc_sections)} unique candidate headings.")
        return toc_sections

    # ------------------------------------------------------------------
    # Step 2: Build final TOC markdown using DeepSeek
    # ------------------------------------------------------------------

    def build_toc_markdown(self, toc_sections: List[str]) -> str:
        if not toc_sections:
            raise ValueError("[TOC] No headings extracted; cannot build TOC.")

        headings_block = "\n".join(f"- {h}" for h in toc_sections)

        final_prompt = (
            "You are organizing a non-fiction book's Table of Contents.\n"
            "Given the following extracted headings, rewrite them into a clean "
            "Markdown table of contents structure.\n\n"
            "Rules:\n"
            "- Use Markdown headings (#, ##, ###).\n"
            "- Do NOT change the wording of any heading.\n"
            "- Group logically, but preserve ordering when possible.\n"
            "- No commentary, only the Markdown TOC.\n\n"
            "Extracted headings:\n\n"
            f"{headings_block}"
        )

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.2,
        )

        toc_md = response.choices[0].message.content or ""
        return toc_md.strip() + "\n"

    # ------------------------------------------------------------------
    # Step 3: Write toc/full.md (only)
    # ------------------------------------------------------------------

    def write_toc_files(self, toc_markdown: str, toc_full_path: str = "toc/full.md") -> None:
        toc_path = Path(toc_full_path)

        # Ensure directory exists
        toc_path.parent.mkdir(parents=True, exist_ok=True)

        # Write only full.md
        toc_path.write_text(toc_markdown, encoding="utf-8")
        print(f"[TOC] Wrote TOC to {toc_path}")

    # ------------------------------------------------------------------

    def generate_from_text(self, full_text: str, toc_full_path: str = "toc/full.md") -> None:
        toc_sections = self.extract_headings(full_text)
        toc_markdown = self.build_toc_markdown(toc_sections)
        self.write_toc_files(toc_markdown, toc_full_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(a.dot(b) / denom)
