from pathlib import Path

from .config import Config
from .embeddings import EmbeddingStore
from .toc_manager import TocManager
from .organizer import ChunkOrganizer, build_prompt_template
from .utils import load_text, split_text_into_chunks
from .toc_generator import TocGenerator  # assuming this is there

# This is the tomeweaver/ folder, no matter where you run `python -m tomeweaver`
BASE_DIR = Path(__file__).resolve().parent


def main():
    # 1. Setup configuration and services
    config = Config.from_env()
    embedding_store = EmbeddingStore(config)

    # 2. Load full input text
    input_path = BASE_DIR / "tmp" / "pizza.txt"
    full_text = load_text(str(input_path))

    # 3. Compute an absolute path for toc/full.md inside the package
    toc_full_path = BASE_DIR / "toc" / "full.md"

    # 3. Generate / refresh TOC from the same text
    toc_generator = TocGenerator(config, embedding_store)
    toc_generator.generate_from_text(
        full_text,
        toc_full_path=str(toc_full_path),
    )

    # 4. Now load the TOC and build the organization prompt
    toc = TocManager(str(toc_full_path))

    # 5. Split the text into content chunks for insertion
    chunks = split_text_into_chunks(full_text, max_chars=10000)
    prompt_template = build_prompt_template(toc)

    # 6. Organize chunks and insert into TOC + embeddings DB
    organizer = ChunkOrganizer(
        client=config.deepseek_client,
        toc=toc,
        embedding_store=embedding_store,
        prompt_template=prompt_template,
    )

    organized_sections = organizer.organize_chunks(chunks)
    organizer.insert_sections(organized_sections)

    print("✅ Generated TOC, updated toc/full.md and embeddings database.")


if __name__ == "__main__":
    main()
