"""
toc_manager.py

Manages:
- Loading and parsing toc/full.md
- Tracking headings
- Inserting new content blocks under specific headings

All paths are resolved relative to the tomeweaver package directory so that
running `python -m tomeweaver` from a parent folder (e.g. `sites/`) still
reads/writes `tomeweaver/toc/full.md`.
"""

from pathlib import Path
from typing import List, Set

# The directory that contains this file (i.e. the tomeweaver package root)
BASE_DIR = Path(__file__).resolve().parent


class TocManager:
    def __init__(self, path: str = "toc/full.md"):
        """
        `path` is treated as relative to the tomeweaver package directory
        unless it is an absolute path.
        """
        raw_path = Path(path)
        if raw_path.is_absolute():
            self.path = raw_path
        else:
            self.path = BASE_DIR / raw_path

        self.lines: List[str] = []
        self._heading_lines: List[str] = []
        self._heading_set: Set[str] = set()

        self.load()

    # ---------- Load + internal cache ----------

    def load(self) -> None:
        """
        Load toc/full.md into memory and build heading caches.
        If the file does not exist yet, initialize an empty document.
        """
        if not self.path.exists():
            # Ensure parent directory exists, but keep the document empty
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.lines = []
            self._heading_lines = []
            self._heading_set = set()
            return

        text = self.path.read_text(encoding="utf-8")
        self.lines = text.splitlines(keepends=True)
        self._refresh_heading_cache()

    def _refresh_heading_cache(self) -> None:
        """
        Rebuild the heading list and set from self.lines.
        """
        self._heading_lines = [
            line.strip()
            for line in self.lines
            if line.strip().startswith("#")
        ]
        self._heading_set = {
            line.lstrip("#").strip()
            for line in self._heading_lines
        }

    # ---------- Heading info ----------

    def get_heading_lines(self) -> List[str]:
        """
        Return all headings (with their leading '#' preserved) as a list.
        """
        return list(self._heading_lines)

    def has_heading(self, heading: str) -> bool:
        """
        Return True if a heading with this text exists (ignoring leading '#').
        """
        return heading in self._heading_set

    # ---------- Insert content ----------

    def insert_block_under_heading(self, heading: str, block: str) -> None:
        """
        Insert `block` immediately after the line that contains the given heading.

        `heading` should be the text without leading '#', matching how headings
        are stored in toc/full.md. Example: "Introduction", not "# Introduction".
        """
        if not block.endswith("\n"):
            block = block + "\n"

        try:
            index = next(
                i
                for i, line in enumerate(self.lines)
                if line.strip().startswith("#")
                and line.lstrip("#").strip() == heading
            )
        except StopIteration:
            print(f"Warning: Heading '{heading}' not found in {self.path}. Skipping.")
            return

        # Insert block right after the heading line
        self.lines.insert(index + 1, block)
        self._refresh_heading_cache()

    # ---------- Save file ----------

    def save(self) -> None:
        """
        Write the current lines back to toc/full.md on disk.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(self.lines), encoding="utf-8")
