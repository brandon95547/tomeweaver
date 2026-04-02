import os
from pathlib import Path

from dotenv import load_dotenv
import openai


BASE_DIR = Path(__file__).resolve().parent


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Config:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.similarity_threshold = _get_float_env("SIMILARITY_THRESHOLD", 0.75)
        self.dimensions = _get_int_env("EMBEDDING_DIMENSIONS", 384)
        self.input_text_path = os.getenv("INPUT_TEXT_PATH", str(BASE_DIR / "tmp" / "pizza.txt"))
        self.toc_full_path = os.getenv("TOC_FULL_PATH", str(BASE_DIR / "toc" / "full.md"))
        self.max_chunk_chars = _get_int_env("MAX_CHUNK_CHARS", 10000)
        self.conservative_mode = os.getenv("CONSERVATIVE_MODE", "false").lower() in ("true", "1", "yes")
        self.catchall_heading = os.getenv("CATCHALL_HEADING", "Miscellaneous")
        self.toc_target_heading_count = _get_int_env("TOC_TARGET_HEADING_COUNT", 60)
        self.content_similarity_threshold = _get_float_env("CONTENT_SIMILARITY_THRESHOLD", 0.95)
        self.overlap_paragraphs = _get_int_env("OVERLAP_PARAGRAPHS", 2)
        self.coverage_threshold = _get_float_env("COVERAGE_THRESHOLD", 0.85)
        self.clean_extracted_text = os.getenv("CLEAN_EXTRACTED_TEXT", "true").lower() in ("true", "1", "yes")
        self.cleanup_chunk_size = _get_int_env("CLEANUP_CHUNK_SIZE", 6000)

    @classmethod
    def from_env(cls):
        return cls()
