from config import Config
from embeddings import EmbeddingStore
from toc_manager import TocManager
from organizer import ChunkOrganizer, build_prompt_template
from utils import load_text, split_text_into_chunks


def main():
    # 1. Setup configuration and services
    config = Config.from_env()
    embedding_store = EmbeddingStore(config)
    toc = TocManager("toc/full.md")

    # 2. Load and split input text
    full_text = load_text("tmp/pizza.txt")
    chunks = split_text_into_chunks(full_text, max_chars=10000)
    print(f"Split into {len(chunks)} chunks.")

    # 3. Build the LLM prompt template based on current TOC
    prompt_template = build_prompt_template(toc)

    # 4. Organize chunks with DeepSeek and insert into TOC + embeddings
    organizer = ChunkOrganizer(
        client=config.deepseek_client,
        toc=toc,
        embedding_store=embedding_store,
        prompt_template=prompt_template,
    )

    organized_sections = organizer.organize_chunks(chunks)
    organizer.insert_sections(organized_sections)

    print("✅ Updated toc/full.md and embeddings database.")


if __name__ == "__main__":
    main()
