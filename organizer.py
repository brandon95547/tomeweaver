"""
organizer.py

Coordinates:
- Building the LLM prompt using the TOC (with stable heading IDs)
- Sending chunks to DeepSeek for organization
- Parsing structured JSON output into heading-based sections
- Using EmbeddingStore + TocManager to dedupe and insert blocks

This version assumes:
- toc/full.md was produced by `toc_generator.py` and contains ONLY headings
  of the form:

    # [H1] Origins of the Theory
    ## [H1.1] Early Online Discussions
    ### [H1.1.1] Specific Forum Threads

- TocManager knows how to:
    - render the outline for prompts (`render_outline_for_prompt()`)
    - insert a markdown block under a heading by ID (`insert_block_under_heading_id()`)

Main public API is unchanged:

    - build_prompt_template(...)
    - ChunkOrganizer(...).organize_chunks(chunks)
    - ChunkOrganizer(...).insert_sections(sections)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .toc_manager import TocManager
from .embeddings import EmbeddingStore

def build_prompt_template(_: Any = None) -> str:
    """
    Build the base prompt template used for each chunk.

    The template MUST contain the placeholders {toc_outline} and {chunk},
    which are filled in later with the actual TOC and text chunk.
    """
    return (
        "You are an organizer, not a writer. Your job is to take messy source text\n"
        "and map it into an EXISTING outline (Table of Contents).\n\n"
        "The outline is given as Markdown headings, each with a stable ID in square\n"
        "brackets, for example:\n"
        "  # [H1] Origins of the Theory\n"
        "  ## [H1.1] Early Online Discussions\n"
        "  ## [H1.2] Historical Parallels\n\n"
        "You MUST NOT invent new heading IDs. You MUST ONLY use the IDs provided.\n"
        "You MUST NOT express any opinion or decide which claim is true.\n"
        "You are allowed to:\n"
        "- Fix grammar and clarity.\n"
        "- Reorder sentences within a block for readability.\n"
        "- Lightly compress or expand text as long as you do not add or remove\n"
        "  factual claims.\n\n"
        "TABLE OF CONTENTS (IDs + headings):\n"
        "{toc_outline}\n"
        "END OF TOC\n\n"
        "SOURCE CHUNK:\n"
        "{chunk}\n"
        "END OF CHUNK\n\n"
        "TASK:\n"
        "1. Decide which heading IDs from the TOC this chunk belongs under.\n"
        "2. For each relevant heading ID, produce a cleaned version of the text that\n"
        "   should appear under that heading.\n"
        "3. If the chunk clearly contains multiple viewpoints or theories under the\n"
        "   same heading, split them into separate entries and give each a short\n"
        "   `subheading` label (e.g. 'Military involvement theory', 'Foreign\n"
        "   intelligence theory').\n"
        "4. Do NOT synthesize your own overall conclusion; just organize and tidy the\n"
        "   existing content.\n\n"
        "OUTPUT FORMAT (VERY IMPORTANT):\n"
        "Return ONLY valid JSON with this exact structure:\n\n"
        "{{\n"
        "  \"sections\": [\n"
        "    {{\n"
        "      \"heading_id\": \"H2.1\",   // one of the IDs from the TOC\n"
        "      \"subheading\": \"Optional label for this viewpoint or angle\",\n"
        "      \"text\": \"Cleaned paragraph(s) that belong under this heading\"\n"
        "    }},\n"
        "    {{\n"
        "      \"heading_id\": \"H2.1\",\n"
        "      \"subheading\": \"Another theory under the same heading\",\n"
        "      \"text\": \"...\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- If the chunk does not clearly belong anywhere, return {{\"sections\": []}}.\n"
        "- Do NOT include any explanation outside the JSON.\n"
        "- Do NOT wrap the JSON in Markdown code fences.\n"
    )

class ChunkOrganizer:
    """
    Orchestrates the second pass:

    - For each text chunk, call DeepSeek with a TOC-aware prompt.
    - Parse JSON to get mappings of chunk -> heading IDs.
    - Insert cleaned blocks under the correct headings in toc/full.md.
    """

    def __init__(
        self,
        client: Any,
        toc: TocManager,
        embedding_store: EmbeddingStore,
        prompt_template: str,
        conservative_mode: bool = False,
        catchall_heading: str = "Miscellaneous",
    ) -> None:
        self.client = client
        self.toc = toc
        self.embedding_store = embedding_store
        self.prompt_template = prompt_template
        self.conservative_mode = conservative_mode
        self.catchall_heading = catchall_heading

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def organize_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Two-pass organizing:
        - Pass 1: Map chunks to existing headings (standard flow).
        - Pass 2: Re-process unmapped content as new sections under catchall heading.

        Returns a flat list of section dicts across all chunks.
        """
        all_sections: List[Dict[str, Any]] = []
        unmapped_chunks: List[str] = []
        conservative = self.conservative_mode

        for idx, chunk in enumerate(chunks, start=1):
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            print(f"[ORG] Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")

            prompt = self._build_prompt(chunk)
            raw_content = self._call_llm(prompt)
            if not raw_content:
                print("  -> Empty response from LLM, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            json_str = self._extract_json(raw_content)
            if not json_str:
                print("  -> Could not locate JSON in LLM response, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"  -> JSON decode error: {e}, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            sections = payload.get("sections") or []
            if not isinstance(sections, list):
                print("  -> 'sections' is not a list; tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            chunk_had_mappings = False
            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                heading_id = (sec.get("heading_id") or "").strip()
                text = (sec.get("text") or "").strip()
                subheading = (sec.get("subheading") or "").strip()

                if not heading_id or not text:
                    continue

                chunk_had_mappings = True
                all_sections.append(
                    {
                        "heading_id": heading_id,
                        "text": text,
                        "subheading": subheading or None,
                    }
                )

            if not chunk_had_mappings:
                unmapped_chunks.append(chunk)

        print(f"[ORG] Pass 1 collected {len(all_sections)} section blocks; {len(unmapped_chunks)} chunk(s) unmapped.")

        if unmapped_chunks:
            print(f"[ORG] Pass 2: Processing {len(unmapped_chunks)} unmapped chunk(s)...")
            unmapped_sections = self._organize_unmapped_chunks(unmapped_chunks)
            all_sections.extend(unmapped_sections)
            print(f"[ORG] Pass 2 added {len(unmapped_sections)} sections from unmapped content.")

        return all_sections

    def _organize_unmapped_chunks(self, unmapped_chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Second pass: organize unmapped content under the catchall heading.
        """
        sections: List[Dict[str, Any]] = []
        catchall = self.catchall_heading

        for chunk in unmapped_chunks:
            chunk = (chunk or "").strip()
            if not chunk:
                continue

            prompt = (
                f"You are organizing content that does not fit the main TOC structure.\n"
                f"Summarize and clean the following chunk into 1-3 short paragraphs suitable for inclusion under '{catchall}'.\n"
                f"Return ONLY the cleaned text, no explanations.\n\n"
                f"Chunk:\n{chunk}"
            )

            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are an expert copy-editor. Clean and organize text without adding facts."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                text = (response.choices[0].message.content or "").strip()
                if text:
                    sections.append(
                        {
                            "heading_id": "CATCHALL",
                            "text": text,
                            "subheading": None,
                        }
                    )
            except Exception as e:
                print(f"  -> Failed to process unmapped chunk: {e}")

        return sections

    def insert_sections(self, sections: List[Dict[str, Any]]) -> None:
        """
        Insert organized sections into toc/full.md, optionally deduplicating
        with FAISS based on conservative_mode config.
        """
        conservative = self.conservative_mode
        catchall = self.catchall_heading
        inserted_count = 0
        skipped_count = 0
        total_chars = 0

        for sec in sections:
            heading_id = sec["heading_id"]
            text = sec["text"]
            subheading = sec.get("subheading")

            # Build the markdown block that will be inserted under this heading.
            block_lines: List[str] = []

            if subheading:
                # Use a level-4 heading for sub-headings, but DO NOT include
                # any heading ID here – IDs are only for the main TOC headings.
                block_lines.append(f"#### {subheading}\n\n")

            block_lines.append(text.strip() + "\n\n")
            block = "".join(block_lines)
            block_chars = len(block)
            total_chars += block_chars

            # Handle catchall sections separately (always insert, no dedup).
            if heading_id == "CATCHALL":
                self.toc.insert_block_under_heading_id(catchall, block)
                inserted_count += 1
                continue

            # For normal sections: apply dedup only if not in conservative mode.
            if conservative:
                # Conservative mode: skip dedup and embeddings, keep all content.
                self.toc.insert_block_under_heading_id(heading_id, block)
                inserted_count += 1
            else:
                emb = self.embedding_store.get_embedding(block)
                if emb is None:
                    print(f"-> Skipping block for {heading_id} due to embedding error.")
                    skipped_count += 1
                    continue

                if self.embedding_store.is_duplicate(emb):
                    print(f"-> Skipping block for {heading_id} (FAISS duplicate).")
                    skipped_count += 1
                    continue

                # Insert into TOC under the given heading ID and register with FAISS.
                self.toc.insert_block_under_heading_id(heading_id, block)
                self.embedding_store.add_block(block, emb)
                inserted_count += 1

        # Persist changes to toc/full.md
        self.toc.save()

        # Validation logging
        mode_info = "(conservative: no dedup)" if conservative else "(dedup enabled)"
        print(f"[ORG] Finished inserting {inserted_count} sections into toc/full.md {mode_info}.")
        if skipped_count > 0:
            print(f"[ORG] Skipped {skipped_count} duplicate blocks.")
        print(f"[ORG] Total content inserted: {total_chars} characters.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, chunk: str) -> str:
        """
        Inject the current TOC (with IDs) and the text chunk into the
        prompt template.
        """
        toc_outline = self.toc.render_outline_for_prompt()
        return self.prompt_template.format(toc_outline=toc_outline, chunk=chunk)

    def _call_llm(self, prompt: str) -> str:
        """
        Make a chat.completions call to DeepSeek and return the raw message text.
        """
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert organizer and copy-editor. "
                        "You NEVER add new facts or personal opinions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        message = response.choices[0].message
        return (getattr(message, "content", None) or "").strip()

    @staticmethod
    def _extract_json(raw: str) -> str | None:
        """
        Try to extract a JSON object from the LLM output.

        The model may sometimes add stray text or Markdown fences, so we:
        - Look for the first '{' and the last '}'.
        - Extract that substring and hope it's valid JSON.

        If nothing plausible is found, return None.
        """
        if not raw:
            return None

        raw_strip = raw.strip()
        if raw_strip.startswith("{") and raw_strip.endswith("}"):
            return raw_strip

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = raw[start : end + 1].strip()
        return candidate or None
