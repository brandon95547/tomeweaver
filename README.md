# TomeWeaver

AI-powered research and book reorganization engine. Takes long-form source material — books, PDFs converted to text, notes — and transforms it into a structured, query-friendly investigation with an AI-generated table of contents, organized section files, and vector embeddings for semantic retrieval.

TomeWeaver is the core engine behind the TomeWeaver product on **Synexis AI**.

---

## Features

- **End-to-end pipeline** — raw text → AI-generated TOC → organized sections → embeddings
- **DeepSeek-powered** — uses DeepSeek for TOC generation, content organization, and text cleanup
- **Semantic chunking** — splits large texts with overlap and deduplicates near-identical headings via embeddings
- **Markdown output** — master `toc/full.md` plus per-heading section files
- **Vector search-ready** — FAISS + SQLite storage for retrieval and question-answering layers
- **Coverage validation** — detects content loss and generates a loss report after each run
- **FastAPI wrapper** — async HTTP API with job queue for integration into SaaS workflows
- **CLI interface** — run as a Python module with optional argument overrides

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| AI Backend | DeepSeek (via OpenAI-compatible client) |
| Embeddings | `sentence-transformers` + FAISS + SQLite |
| API Server | FastAPI + Uvicorn |
| Config | `python-dotenv` |

---

## Project Structure

```text
tomeweaver/
├── __init__.py
├── __main__.py          # Module entry point → calls main()
├── main.py              # Pipeline orchestrator
├── api.py               # FastAPI HTTP wrapper
├── config.py            # Environment + configuration loader
├── embeddings.py        # FAISS + SQLite vector store
├── toc_generator.py     # Builds TOC from raw text via DeepSeek
├── toc_manager.py       # Reads/manipulates toc/full.md
├── organizer.py         # Chunk → section organizer + prompt builder
├── text_cleaner.py      # AI-powered OCR / text artifact cleanup
├── utils.py             # I/O helpers, chunking, splitting
├── validation.py        # Coverage scoring + loss report generation
├── Makefile             # Dev convenience targets
├── requirements.txt
├── data/
│   └── input/           # Raw input text files
└── toc/
    └── full.md          # Generated master table of contents
```

---

## Prerequisites

- Python **3.10+** (3.11+ recommended)
- `pip` and `venv` (or `python3 -m venv`)
- A **DeepSeek API key**
- Basic build tools for FAISS (if no precompiled wheel is available for your platform)

---

## Environment Variables

Copy the example and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | ✅ | — | DeepSeek API key |
| `DEEPSEEK_BASE_URL` | — | `https://api.deepseek.com` | DeepSeek API base URL |
| `INPUT_TEXT_PATH` | — | `./tmp/pizza.txt` | Path to the source text file |
| `TOC_FULL_PATH` | — | `./toc/full.md` | Output path for the master TOC |
| `MAX_CHUNK_CHARS` | — | `10000` | Max characters per content chunk |
| `SIMILARITY_THRESHOLD` | — | `0.75` | Cosine similarity threshold for heading dedup |
| `CONTENT_SIMILARITY_THRESHOLD` | — | `0.95` | Threshold for content-level dedup |
| `EMBEDDING_DIMENSIONS` | — | `384` | Embedding vector dimensions |
| `TOC_TARGET_HEADING_COUNT` | — | `60` | Target number of TOC headings |
| `OVERLAP_PARAGRAPHS` | — | `2` | Paragraph overlap between chunks |
| `COVERAGE_THRESHOLD` | — | `0.85` | Minimum acceptable coverage score |
| `CONSERVATIVE_MODE` | — | `false` | Stricter section matching |
| `CATCHALL_HEADING` | — | `Miscellaneous` | Heading for unmatched content |
| `CLEAN_EXTRACTED_TEXT` | — | `true` | Run AI text cleanup before processing |
| `CLEANUP_CHUNK_SIZE` | — | `6000` | Chunk size for text cleanup pass |

---

## Getting Started — Development

### 1. Create and activate a virtual environment

```bash
cd tomeweaver
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or use the Makefile shortcut (creates `.venv` and installs in one step):

```bash
make install
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY at minimum
```

### 4. Run the pipeline (CLI)

```bash
python -m tomeweaver
```

With optional overrides:

```bash
python -m tomeweaver \
  --input ./data/input/source.txt \
  --toc ./toc/full.md \
  --max-chars 12000
```

### 5. Run the API server (development)

```bash
uvicorn tomeweaver.api:app --reload --host 0.0.0.0 --port 8010
```

The API is available at `http://localhost:8010`. Verify with:

```bash
curl http://localhost:8010/health
```

---

## Getting Started — Production

### 1. Set up the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

Ensure `.env` is present with production values (or export variables directly):

```bash
export DEEPSEEK_API_KEY=your_production_key
```

### 3. Run the API server

```bash
uvicorn tomeweaver.api:app \
  --host 0.0.0.0 \
  --port 8010 \
  --workers 2
```

For deployment behind a reverse proxy (Nginx, Caddy), bind to `127.0.0.1` instead:

```bash
uvicorn tomeweaver.api:app \
  --host 127.0.0.1 \
  --port 8010 \
  --workers 2
```

> **Tip:** Use a process manager like `systemd` or `supervisord` to keep the server running.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (returns active job count) |
| `POST` | `/run` | Start a pipeline job (async, returns `job_id`) |
| `GET` | `/jobs/{job_id}` | Check job status |
| `GET` | `/jobs/{job_id}/result` | Retrieve completed job result |

### POST /run — Request Body

```json
{
  "text": "Full source text here...",
  "input_path": "/path/to/file.txt",
  "toc_full_path": "/path/to/output/toc.md",
  "max_chunk_chars": 10000
}
```

Provide either `text` (inline) or `input_path` (file on disk). The job runs asynchronously — poll `/jobs/{job_id}` for status.

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Create `.venv` and install dependencies |
| `make run` | Run the TomeWeaver pipeline |
| `make clean` | Remove `__pycache__` and `.pyc` files |
| `make clean_venv` | Delete the virtual environment |
| `make reinstall` | Clean everything and reinstall from scratch |

---

## License

See [LICENSE](LICENSE).