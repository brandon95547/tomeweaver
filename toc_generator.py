# toc_generator.py

"""
toc_generator.py

Builds a candidate Table of Contents (TOC) from raw book text using DeepSeek + local
embeddings, and writes ONLY:
- toc/full.md

Changes vs previous version:
- Adds a second-pass dedupe after DeepSeek's "organized TOC" step.
- Normalizes headings to avoid near-duplicates ("Origin of the Theory" vs
  "Origins of this Theory", etc.).
- Assigns stable hierarchical IDs like [H1], [H1.2], [H1.2.1] to every heading.
- Ensures toc/full.md is structure-only (headings only; no bullets or prose).
"""

from pathlib import Path
from typing import List, Set, Tuple

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
            "describe the the main topics.\n\n"
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
                line = line.strip()
                if not line:
                    continue

                # Expect something like "# Heading Title" or "## Subheading"
                m = re.match(r"^(#{1,3})\s+(.*)$", line)
                if not m:
                    continue

                hashes, title = m.groups()
                title = title.strip()
                heading_text = f"{hashes} {title}"

                # Embedding-based dedupe across ALL chunks
                emb = self.embedding_store.get_embedding(heading_text)
                if emb is None:
                    continue

                if any(
                    self._cosine_similarity(emb, prev) >= self.similarity_threshold
                    for prev in seen_embeddings
                ):
                    # Too similar to an existing heading
                    continue

                if heading_text not in heading_texts:
                    heading_texts.add(heading_text)
                    seen_embeddings.append(emb)
                    toc_sections.append(heading_text)

        print(f"[TOC] Collected {len(toc_sections)} unique candidate headings.")
        return toc_sections

    # ------------------------------------------------------------------
    # Step 2: Build final TOC markdown using DeepSeek, then dedupe + ID
    # ------------------------------------------------------------------

    def build_toc_markdown(self, toc_sections: List[str]) -> str:
        if not toc_sections:
            raise ValueError("[TOC] No headings extracted; cannot build TOC.")

        # Feed the raw headings into DeepSeek to organize into a logical TOC.
        headings_block = "\n".join(f"- {h}" for h in toc_sections)

        final_prompt = (
            "You are organizing a non-fiction book's Table of Contents.\n"
            "Given the following extracted headings, rewrite them into a clean "
            "Markdown table of contents structure.\n\n"
            "Rules:\n"
            "- Use Markdown headings only (#, ##, ###).\n"
            "- Do NOT change the wording of any heading text, only group and order.\n"
            "- Do NOT add bullet lists or explanatory paragraphs.\n"
            "- No commentary, only Markdown headings.\n\n"
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

        toc_md_raw = (response.choices[0].message.content or "").strip()
        if not toc_md_raw:
            raise ValueError("[TOC] DeepSeek returned empty TOC markdown.")

        # Parse the returned markdown TOC and:
        #  - keep only heading lines (#, ##, ###)
        #  - dedupe again using normalized text + embeddings
        #  - assign stable hierarchical IDs [H1], [H1.2], [H1.2.1]
        final_headings: List[Tuple[str, str]] = []  # (hashes, title)
        seen_norms: Set[str] = set()
        seen_embs: List[np.ndarray] = []

        for line in toc_md_raw.splitlines():
            line = line.rstrip()
            if not line:
                continue

            m = re.match(r"^(#{1,6})\s+(.*\S.*)$", line)
            if not m:
                # Ignore bullets or any non-heading content
                continue

            hashes, title = m.groups()
            title = title.strip()
            heading_text_for_emb = f"{hashes} {title}"

            norm = self._normalize_heading_text(title)
            if norm in seen_norms:
                # Already saw a semantically identical heading
                continue

            emb = self.embedding_store.get_embedding(heading_text_for_emb)
            if emb is None:
                continue

            if any(
                self._cosine_similarity(emb, prev) >= self.similarity_threshold
                for prev in seen_embs
            ):
                # Embedding-based near-duplicate
                continue

            seen_norms.add(norm)
            seen_embs.append(emb)
            final_headings.append((hashes, title))

        print(f"[TOC] After organization+dedupe: {len(final_headings)} headings.")

        # Assign hierarchical IDs and build final markdown
        toc_with_ids_lines: List[str] = []
        h1_idx = 0
        h2_idx = 0
        h3_idx = 0

        for hashes, title in final_headings:
            level = len(hashes)

            if level == 1:
                h1_idx += 1
                h2_idx = 0
                h3_idx = 0
                hid = f"H{h1_idx}"
            elif level == 2:
                if h1_idx == 0:
                    h1_idx = 1
                h2_idx += 1
                h3_idx = 0
                hid = f"H{h1_idx}.{h2_idx}"
            elif level == 3:
                if h1_idx == 0:
                    h1_idx = 1
                if h2_idx == 0:
                    h2_idx = 1
                h3_idx += 1
                hid = f"H{h1_idx}.{h2_idx}.{h3_idx}"
            else:
                # Fallback: treat anything deeper as a 3rd-level child
                if h1_idx == 0:
                    h1_idx = 1
                if h2_idx == 0:
                    h2_idx = 1
                h3_idx += 1
                hid = f"H{h1_idx}.{h2_idx}.{h3_idx}"

            # Final heading line with ID, e.g.:
            #   # [H1] Origins of the Theory
            #   ## [H1.1] Early Accounts
            line_with_id = f"{hashes} [{hid}] {title}"
            toc_with_ids_lines.append(line_with_id)

        # Ensure final file ends with a newline
        return "\n".join(toc_with_ids_lines).strip() + "\n"

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

    @staticmethod
    def _normalize_heading_text(text: str) -> str:
        """
        Normalize heading text to detect duplicates:
        - Lowercase
        - Remove punctuation
        - Drop common stopwords
        - Collapse whitespace

        Example:
            "Origins of the Theory" ->
            "origins theory"
        """
        text = text.lower()
        # Remove brackets/IDs if they slipped in for some reason
        text = re.sub(r"\[[^\]]+\]", " ", text)
        # Remove punctuation-ish characters
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()

        stopwords = {
            "the",
            "a",
            "an",
            "of",
            "about",
            "this",
            "that",
            "in",
            "on",
            "for",
            "to",
            "and",
            "with",
            "from",
            "into",
            "introduction",
            "chapter",
            "section",
            "part",
        }

        tokens = [t for t in tokens if t not in stopwords]
        return " ".join(tokens)
