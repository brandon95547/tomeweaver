# 📚 Tome Weaver Organizer Pipeline

*A modular Python system for processing large text files, organizing content into a structured Table of Contents, deduplicating via embeddings, and updating a master markdown document.*

---

## 🚀 Overview

This project takes a large raw text file (ex: book content, scanned OCR text, research dumps), breaks it into chunks, sends each chunk to an LLM with a Table of Contents-aware prompt, and inserts the structured content back into `toc/full.md`.

All inserted blocks are embedded with Hugging Face sentence-transformers, deduplicated using FAISS, and stored in SQLite.

The result:
✔️ Cleanly organized, structured markdown
✔️ No duplicate sections
✔️ Fully automated TOC-guided book building

---

## 🧱 Project Structure

```
project/
│
├── main.py
├── config.py
├── embeddings.py
├── toc_manager.py
├── organizer.py
├── utils.py
│
└── toc/
    └── full.md
```

Each module is responsible for one clean, isolated part of the workflow.

---

## 🔧 Core Files and Responsibilities

### **1. config.py**

Loads environment variables, initializes:

* DeepSeek/OpenAI client
* Hugging Face endpoint + keys
* Global config values (dimensions, threshold)

This file defines:

```python
Config
```

Used across all other modules.

---

### **2. embeddings.py**

Handles all embedding work:

* SQLite table creation
* FAISS index building
* Hugging Face embedding API calls
* Duplicate detection
* Adding new blocks into DB + FAISS

Defines:

```python
EmbeddingStore
```

---

### **3. toc_manager.py**

Manages `toc/full.md`:

* Loads file
* Extracts headings
* Checks if a heading exists
* Inserts content directly under specific TOC headings
* Saves the updated file

Defines:

```python
TocManager
```

---

### **4. organizer.py**

This is the "brain" of the pipeline:

* Builds the LLM prompt using the existing TOC
* Sends each chunk of text to DeepSeek
* Parses returned markdown headings
* For each heading:

  * Validates heading against ToC
  * Embeds the block
  * Deduplicates via FAISS
  * Inserts into TOC + DB

Defines:

```python
ChunkOrganizer
build_prompt_template()
```

---

### **5. utils.py**

Pure helper functions:

* `hash_text()`
* `serialize_vector()`
* `deserialize_vector()`
* `load_text()`
* `split_text_into_chunks()`

No dependencies, used by multiple modules.

---

## 🏃‍♂️ Execution Flow

Below is the **full end-to-end pipeline**, exactly how the script runs.

### **1. Load environment + create core services**

```python
config = Config.from_env()
embedding_store = EmbeddingStore(config)
toc = TocManager("toc/full.md")
```

➡️ DeepSeek client initialized
➡️ Hugging Face ready
➡️ SQLite table ensured
➡️ FAISS loaded with existing embeddings
➡️ TOC headings loaded

---

### **2. Load & split input text**

```python
full_text = load_text("tmp/pizza.txt")
chunks = split_text_into_chunks(full_text, max_chars=10000)
```

➡️ Long text broken into manageable chunks
➡️ Clean chunk boundaries (sentence → newline → hard cut)

---

### **3. Build prompt based on the current TOC**

```python
prompt_template = build_prompt_template(toc)
```

Prompt contains:

* Full list of TOC headings
* Editor instructions
* Markdown output requirements
* `{chunk}` placeholder

---

### **4. Send each chunk to DeepSeek LLM**

```python
organizer = ChunkOrganizer(config.deepseek_client, toc, embedding_store, prompt_template)
organized_sections = organizer.organize_chunks(chunks)
```

DeepSeek returns structured markdown for each chunk:

```
# Part Title
- Important bullet
- Another detail

# Another Heading
Paragraphs...
```

---

### **5. Parse & insert structured sections**

```python
organizer.insert_sections(organized_sections)
```

For each organized section:

1. Detect headings
2. Ensure heading exists in TOC
3. Collect block text
4. Embed via Hugging Face
5. Use FAISS to detect duplicates
6. If unique:

   * Insert into TOC markdown under the correct heading
   * Insert embedding into SQLite
   * Add vector to FAISS

The updated `toc/full.md` is saved at the end.

---

## 🔄 High-Level Flowchart

```
Load Config → Init Embeddings → Init TOC
        ↓
Load Input Text → Split into Chunks
        ↓
Build Prompt Template
        ↓
[ For Each Chunk ]
    → Send to DeepSeek
    → Receive structured markdown
        ↓
Parse Organized Output
Validate Heading
Get Embedding
Check Duplicate
Insert Block (TOC + DB + FAISS)
        ↓
Save Updated toc/full.md
```

---

## 🧪 Running the Pipeline

Make sure your `.env` contains:

```
DEEPSEEK_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
```

Then run:

```
python3 main.py
```

---

## ✔️ Output

* Updated `toc/full.md` with new content under correct headings
* `embeddings.db` populated with unique content
* `faiss.index` updated in-memory
* No duplicate sections added

---

## ❤️ Summary

This structure gives you:

* Maximum clarity
* Full modularity
* Easy debugging
* Clean separation of concerns
* A readable `main.py` that tells the full story

If you want, I can also add:

* Logging instead of print statements
* Progress bars
* CLI arguments (`--input`, `--toc`, etc.)
* Unit tests