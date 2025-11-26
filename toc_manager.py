"""
toc_manager.py

Manages:
- Loading and parsing toc/full.md
- Tracking headings
- Inserting new content blocks under specific headings
"""

from pathlib import Path
from typing import List, Set


class TocManager:
    def __init__(self, path: str):
        self.path = Path(path)
        # Keep line endings so we can safely re-join later
        text = self.path.read_text(encoding="utf-8")
        self.lines = text.splitlines(keepends=True)

        self._heading_lines: List[str] = [
            line.strip()
            for line in self.lines
            if line.strip().startswith("#")
        ]
        self._heading_set: Set[str] = set(
            line.lstrip("#").strip() for line in self._heading_lines
        )

    # ---------- Heading info ----------

    def get_heading_lines(self) -> List[str]:
        """
        Return all markdown heading lines (e.g., "# Part 1", "## Chapter 2").
        """
        return self._heading_lines

    def has_heading(self, heading: str) -> bool:
        """
        Check if a normalized heading text exists in the current TOC,
        ignoring leading '#'.
        """
        return heading in self._heading_set

    # ---------- Insert content ----------

    def insert_block_under_heading(self, heading: str, block: str):
        """
        Insert the given block (string with newlines) immediately after
        the specified heading line.
        """
        try:
            index = next(
                i
                for i, line in enumerate(self.lines)
                if line.lstrip("#").strip() == heading
            )
            self.lines.insert(index + 1, block)
        except StopIteration:
            print(f"Warning: Heading '{heading}' not found in {self.path}. Skipping.")

    # ---------- Save file ----------

    def save(self):
        """
        Write the current lines back to toc/full.md.
        """
        self.path.write_text("".join(self.lines), encoding="utf-8")
