# toc_generator.py

"""
toc_generator.py

Builds a candidate TOC from raw book text using DeepSeek + embeddings,
and writes:
- toc/full.md
- one .md file per top-level heading in ./toc
"""

from pathlib import Path
from typing import List, Set

import numpy as np
import re

from .config import Config
from .embeddings import EmbeddingStore
from .utils import split_text_into_chunks


class TocGenerator:
    def __init__(self, config: Config, embedding_store: EmbeddingStore):
        self.config = config
        self.embedding_store = embedding_store
        self.client = config.deepseek_client
        self.similarity_threshold = config.similarity_threshold

    # ---------- Step 1: Extract candidate headings from chunks ----------

    def extract_headings(
        self,
        full_text: str,
        max_chars_per_chunk: int = 10_000,
    ) -> List[str]:
        """
        Use DeepSeek to extract candidate headings from each text chunk,
        dedupe them using embeddings, and return a flat list of unique headings.
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
                temperature=0.3,
            )

            toc_output = response.choices[0].message.content or ""
            for line in toc_output.splitlines():
                stripped = line.strip()
                if not stripped or not stripped.startswith("#"):
                    continue

                # Strip hashes for similarity and uniqueness checks
                heading_text = stripped.lstrip("#").strip()
                if not heading_text:
                    continue

                emb = self.embedding_store.get_embedding(heading_text)
                if emb is None:
                    continue

                if any(
                    self._cosine_similarity(emb, seen) > self.similarity_threshold
                    for seen in seen_embeddings
                ):
                    # Very similar heading already seen
                    continue

                if heading_text not in heading_texts:
                    heading_texts.add(heading_text)
                    seen_embeddings.append(emb)
                    toc_sections.append(heading_text)

        print(f"[TOC] Collected {len(toc_sections)} unique candidate headings.")
        return toc_sections

    # ---------- Step 2: Ask DeepSeek to build the final TOC markdown ----------

    def build_toc_markdown(self, toc_sections: List[str]) -> str:
        if not toc_sections:
            raise ValueError("[TOC] No headings extracted; cannot build TOC.")

        headings_block = "\n".join(f"- {h}" for h in toc_sections)

        final_prompt = (
            "You are a professional book editor.\n"
            "Organize the following list of headings into a complete and coherent "
            "Table of Contents for a structured investigation.\n\n"
            "Rules:\n"
            "- Use only Markdown headings with '#', '##', and '###'.\n"
            "- Group similar topics under logical Parts and Sections.\n"
            "- Do NOT change the wording of the headings.\n"
            "- Each heading should appear only once (no duplicates).\n"
            "- Preserve the original order as much as possible, but you may group "
            "and reorder for clarity.\n\n"
            "Here are the extracted headings:\n\n"
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

    # ---------- Step 3: Write toc/full.md and per-heading .md files ----------

    def write_toc_files(self, toc_markdown: str, toc_full_path: str = "toc/full.md") -> None:
        toc_path = Path(toc_full_path)
        toc_dir = toc_path.parent
        toc_dir.mkdir(parents=True, exist_ok=True)

        toc_path.write_text(toc_markdown, encoding="utf-8")
        print(f"[TOC] Wrote full TOC to {toc_path}")

        current_title = None
        section_lines: List[str] = []

        for line in toc_markdown.splitlines():
            if line.strip().startswith("#"):
                # Save the previous section
                if current_title and section_lines:
                    filename = self._slugify_heading(current_title)
                    (toc_dir / f"{filename}.md").write_text(
                        "\n".join(section_lines) + "\n", encoding="utf-8"
                    )

                current_title = line
                section_lines = [line]
            elif current_title:
                section_lines.append(line)

        # Save the last section
        if current_title and section_lines:
            filename = self._slugify_heading(current_title)
            (toc_dir / f"{filename}.md").write_text(
                "\n".join(section_lines) + "\n", encoding="utf-8"
            )

        print(f"[TOC] Individual TOC files saved in {toc_dir}")

    # ---------- Orchestrator ----------

    def generate_from_text(self, full_text: str, toc_full_path: str = "toc/full.md") -> None:
        toc_sections = self.extract_headings(full_text)
        toc_markdown = self.build_toc_markdown(toc_sections)
        self.write_toc_files(toc_markdown, toc_full_path)

    # ---------- Helpers ----------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(a.dot(b) / denom)

    @staticmethod
    def _slugify_heading(heading_line: str) -> str:
        title = heading_line.lstrip("#").strip()
        return re.sub(r"[^a-zA-Z0-9_-]+", "_", title).lower()
