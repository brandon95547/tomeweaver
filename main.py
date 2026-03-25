import argparse
from pathlib import Path
from typing import Optional

from .config import Config
from .embeddings import EmbeddingStore
from .toc_manager import TocManager
from .organizer import ChunkOrganizer, build_prompt_template
from .utils import load_text, split_text_into_chunks
from .toc_generator import TocGenerator  # assuming this is there

# This is the tomeweaver/ folder, no matter where you run `python -m tomeweaver`
BASE_DIR = Path(__file__).resolve().parent


def run_pipeline(
    input_path: Optional[str] = None,
    toc_full_path: Optional[str] = None,
    max_chunk_chars: Optional[int] = None,
) -> dict:
    # 1. Setup configuration and services
    config = Config.from_env()
    embedding_store = EmbeddingStore(config)

    # 2. Load full input text
    resolved_input_path = Path(input_path or config.input_text_path)
    full_text = load_text(str(resolved_input_path))

    # 3. TOC output path
    resolved_toc_path = Path(toc_full_path or config.toc_full_path)

    # 3. Generate / refresh TOC from the same text
    toc_generator = TocGenerator(config, embedding_store)
    toc_generator.generate_from_text(
        full_text,
        toc_full_path=str(resolved_toc_path),
    )

    # 4. Now load the TOC and build the organization prompt
    toc = TocManager(str(resolved_toc_path))

    # 5. Split the text into content chunks for insertion
    resolved_max_chunk_chars = int(max_chunk_chars or config.max_chunk_chars)
    chunks = split_text_into_chunks(full_text, max_chars=resolved_max_chunk_chars)
    prompt_template = build_prompt_template(toc)

    # 6. Organize chunks and insert into TOC + embeddings DB
    organizer = ChunkOrganizer(
        client=config.deepseek_client,
        toc=toc,
        embedding_store=embedding_store,
        prompt_template=prompt_template,
        conservative_mode=config.conservative_mode,
        catchall_heading=config.catchall_heading,
    )

    organized_sections = organizer.organize_chunks(chunks)
    organizer.insert_sections(organized_sections)

    return {
        "ok": True,
        "input_path": str(resolved_input_path),
        "toc_full_path": str(resolved_toc_path),
        "chunk_count": len(chunks),
        "organized_section_count": len(organized_sections),
    }


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Run TomeWeaver pipeline.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input text file path. Defaults to INPUT_TEXT_PATH from env.",
    )
    parser.add_argument(
        "--toc",
        default=None,
        help="Output TOC markdown path. Defaults to TOC_FULL_PATH from env.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Max chars per content chunk. Defaults to MAX_CHUNK_CHARS from env.",
    )
    args = parser.parse_args(argv)

    result = run_pipeline(
        input_path=args.input,
        toc_full_path=args.toc,
        max_chunk_chars=args.max_chars,
    )
    print(
        "✅ Generated TOC and updated embeddings "
        f"(chunks={result['chunk_count']}, sections={result['organized_section_count']})."
    )


if __name__ == "__main__":
    main()
