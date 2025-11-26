"""
organizer.py

Coordinates:
- Building the LLM prompt using the TOC
- Sending chunks to DeepSeek for organization
- Parsing LLM output into heading-based sections
- Using EmbeddingStore + TocManager to dedupe and insert blocks
"""

import re
from typing import List, Any

from .toc_manager import TocManager
from .embeddings import EmbeddingStore


# ---------- Prompt builder ----------

def build_prompt_template(toc: TocManager) -> str:
    """
    Build the prompt template using the existing TOC headings.
    Matches the behavior of the original inline prompt_template.
    """
    toc_headings_list = toc.get_heading_lines()

    return (
        "You are a professional book editor working with the following existing Table of Contents:\n"
        + "\n".join(toc_headings_list)
        + "\n\nYour task is to read the provided raw text and do the following:\n"
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


# ---------- ChunkOrganizer ----------

class ChunkOrganizer:
    def __init__(
        self,
        client: Any,
        toc: TocManager,
        embedding_store: EmbeddingStore,
        prompt_template: str,
    ):
        """
        client: DeepSeek/OpenAI-compatible client instance
        toc: TocManager instance
        embedding_store: EmbeddingStore instance
        prompt_template: format-string with {chunk} placeholder
        """
        self.client = client
        self.toc = toc
        self.embedding_store = embedding_store
        self.prompt_template = prompt_template
        self.heading_pattern = re.compile(r"^(#+)\s+(.*)")

    # ---------- LLM step ----------

    def organize_chunks(self, chunks: List[str]) -> List[str]:
        """
        Send each chunk to DeepSeek and collect the organized markdown responses.
        """
        organized_sections: List[str] = []

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}...")

            prompt = self.prompt_template.format(chunk=chunk)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            organized_output = response.choices[0].message.content
            organized_sections.append(organized_output)

        return organized_sections

    # ---------- Insert sections into TOC + embeddings ----------

    def insert_sections(self, organized_sections: List[str]):
        for section in organized_sections:
            self._process_section(section)

        # After all sections processed, save TOC once
        self.toc.save()

    def _process_section(self, section: str):
        lines = section.strip().splitlines()
        i = 0

        while i < len(lines):
            match = self.heading_pattern.match(lines[i])
            if not match:
                i += 1
                continue

            _, heading_text = match.groups()
            normalized_heading = heading_text.strip()

            # Collect all lines under this heading until the next heading
            content_lines = []
            i += 1
            while i < len(lines) and not self.heading_pattern.match(lines[i]):
                content_lines.append(lines[i])
                i += 1

            # Verify heading exists in original ToC
            if not self.toc.has_heading(normalized_heading):
                print(
                    f"Warning: Heading '{normalized_heading}' not found in toc/full.md. Skipping."
                )
                continue

            # Build block text exactly like the original script
            new_block = "\n" + "\n".join(content_lines).strip() + "\n"
            print(f"new_block -- {new_block}")

            # Get embedding
            emb = self.embedding_store.get_embedding(new_block)
            if emb is None:
                print("-> Skipping new_block due to embedding error.")
                continue

            # Duplicate check via FAISS
            if self.embedding_store.is_duplicate(emb):
                print(
                    f"-> Skipping new_block under '{normalized_heading}' (FAISS duplicate)"
                )
                continue

            # Insert into TOC + DB/FAISS
            self.toc.insert_block_under_heading(normalized_heading, new_block)
            self.embedding_store.add_block(new_block, emb)
