# TomeWeaver

TomeWeaver is a research/book reorganization engine designed to take long-form source material (books, PDFs converted to text, notes, etc.) and transform it into a structured, query-friendly "investigation" with an AI-generated table of contents (TOC), section files, and embeddings for retrieval.

It is the core engine that will power the TomeWeaver SaaS on **Synexis AI**.

---

## Features

- 🔁 **End-to-end pipeline**: from raw text → AI-generated TOC → organized sections → embeddings.
- 🧠 **DeepSeek-powered**: uses DeepSeek for TOC generation and content organization.
- 🧩 **Chunking & deduplication**: splits large text into chunks and avoids near-duplicate headings using embeddings.
- 📚 **Markdown output**:
  - `toc/full.md` — master table of contents.
  - Per-heading Markdown files for each top-level section.
- 🔎 **Embeddings & search-ready**:
  - Uses FAISS + SQLite for vector storage.
  - Ready for retrieval / question-answering layers on top.
- 🧱 **Modular architecture** (config, embeddings, TOC generation, organization, utilities).
- 🚀 **Package-first design**: runs as a Python module via `python -m tomeweaver`.

---

## Project Structure

_(Adjust this to match your actual layout if needed.)_

```text
tomeweaver/
  __init__.py
  __main__.py         # Module entrypoint → calls main()
  main.py             # Orchestrates the full pipeline

  config.py           # Config & environment loading (Config)
  embeddings.py       # EmbeddingStore (FAISS + SQLite integration)
  toc_manager.py      # TocManager for reading/manipulating toc/full.md
  toc_generator.py    # TocGenerator for building TOC from raw text
  organizer.py        # ChunkOrganizer & prompt building logic
  utils.py            # Helper utilities (I/O, chunking, etc.)

data/
  input/              # Raw input text files (e.g. source.txt)
  output/             # Optional: organized output (if used)
toc/
  full.md             # Generated master TOC
  *.md                # Per-heading TOC files

requirements.txt
.env.example
README.md
````

---

## Requirements

* Python **3.10+** (recommended 3.11+)
* `pip` and `virtualenv` (or `python -m venv`)
* DeepSeek API key
* Basic build tools for FAISS (if not using a precompiled wheel)

Core Python dependencies (in `requirements.txt`), e.g.:

* `openai`
* `python-dotenv`
* `numpy`
* `faiss-cpu`
* `sqlite3` (standard library)
* Any other libraries you’ve added

---

## Installation

From the project root:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows (PowerShell/CMD)

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Configuration

TomeWeaver uses a `.env` file loaded via `python-dotenv`.

Create your `.env` (or copy from the example):

```bash
cp .env.example .env
```

Then set the required variables inside `.env`:

```env
# DeepSeek
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY

# Optional / project-specific settings:
# Path to SQLite DB (or leave default if handled in Config)
SQLITE_DB_PATH=./tomeweaver.db

# Embedding model identifier (if your EmbeddingStore uses this)
EMBEDDING_MODEL=text-embedding-3-small

# Input text file (can be overridden by Config logic)
INPUT_TEXT_PATH=./data/input/source.txt

# Similarity threshold for heading deduplication
SIMILARITY_THRESHOLD=0.85
```

The `Config` class in `config.py` is responsible for:

* Reading the environment variables.
* Initializing the DeepSeek client:

  * `openai.OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")`
* Setting defaults for input paths, DB paths, thresholds, etc.

---

## How to Run TomeWeaver

TomeWeaver is designed to run as a Python module.

From the project root (where the `tomeweaver/` package directory lives):

```bash
#when this is not created
cd sites/tomeweaver
python3 -m venv venv
cd ..
source tomeweaver/venv/bin/activate
pip install -r tomeweaver/requirements.txt
python -m tomeweaver


```

This will:

1. Load configuration and environment variables.
2. Initialize the embeddings store (SQLite + FAISS).
3. Load the raw input text (e.g. from `INPUT_TEXT_PATH`).
4. Generate or refresh the table of contents (TOC).
5. Split the text into chunks.
6. Organize chunks into sections aligned to the TOC using DeepSeek.
7. Persist organized content and embeddings.

---

## Pipeline / Data Flow

Here’s the high-level flow implemented in `main.py` and the supporting classes:

1. **Bootstrap configuration and services**

   * `Config.from_env()` reads `.env` and sets up:

     * DeepSeek client
     * Paths
     * Similarity thresholds
   * `EmbeddingStore` initializes FAISS + SQLite.

2. **Load raw source text**

   * `utils.load_text(INPUT_TEXT_PATH)` loads the main input file (e.g., combined book text).

3. **Generate / refresh TOC**

   * `TocGenerator`:

     * Splits the full text into manageable chunks.
     * Uses DeepSeek to propose candidate headings per chunk.
     * Converts each heading into embeddings, deduplicates similar ones using cosine similarity.
     * Builds a coherent Markdown TOC using DeepSeek again.
     * Writes:

       * `toc/full.md` (master TOC)
       * Individual per-heading TOC files in `./toc`.

4. **Load TOC for organization**

   * `TocManager("toc/full.md")` parses the Markdown TOC and provides an interface for matching/attaching content.

5. **Chunk the source text**

   * `utils.split_text_into_chunks(full_text, max_chars=...)` splits the full text into working units for AI calls.

6. **Organize chunks into sections**

   * `ChunkOrganizer`:

     * Builds prompts using `build_prompt_template(toc)` that give DeepSeek the TOC context.
     * Sends each chunk to DeepSeek with instructions on:

       * Which headings it relates to.
       * How to position content in the investigation.
     * Returns a structured list of "organized sections".

7. **Insert sections + embeddings**

   * `ChunkOrganizer.insert_sections(organized_sections)`:

     * Writes organized content out (Markdown or into DB, depending on your implementation).
     * Creates and stores embeddings linked to section IDs / titles in FAISS + SQLite.

---

## Regenerating the TOC

By default, the main pipeline calls the TOC generator as part of the flow.

If you maintain `generate_toc.py` as a lightweight wrapper, you can regenerate the TOC alone:

```bash
python generate_toc.py
```

But in the integrated architecture, the recommended way is simply:

```bash
python -m tomeweaver
```

You can later add a flag (e.g. `--regen-toc` or `--skip-toc`) via a CLI layer, but that’s outside the core engine.

---

## Development Notes

* **Package entrypoint**:
  `tomeweaver/__main__.py` should look roughly like:

  ```python
  from .main import main

  if __name__ == "__main__":
      main()
  ```

* **Main orchestrator**:
  `main()` in `main.py` wires together:

  * `Config`
  * `EmbeddingStore`
  * `TocGenerator`
  * `TocManager`
  * `ChunkOrganizer`
  * Utility functions (load text, splitting, etc.)

* **Extensibility ideas**:

  * Add CLI arguments (e.g. input path, output path, toggle TOC regeneration).
  * Support multiple input documents.
  * Export JSON metadata for sections to feed into your SaaS frontend.

---

## Using TomeWeaver in a SaaS Context

For your Synexis AI platform, TomeWeaver can be wrapped in:

* A **FastAPI / Flask / Django** HTTP API that:

  * Accepts uploads or text.
  * Triggers the TomeWeaver pipeline.
  * Stores results in a shared DB / object storage.
* A **task queue** (RQ, Celery, etc.) for long-running jobs.
* A **frontend** that:

  * Displays the generated TOC.
  * Lets users navigate through sections.
  * Performs semantic search using stored embeddings.

This README documents the **core engine**. The SaaS wrapper can be described in a separate `docs/` section later.

---

## License

*Add your chosen license here (MIT, proprietary, etc.).*