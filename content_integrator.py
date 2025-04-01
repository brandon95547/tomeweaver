# Import required libraries
import os  # for file and directory operations
import sqlite3  # for interacting with a SQLite database
import numpy as np  # for numerical and array operations
import faiss  # Facebook AI Similarity Search, used for efficient vector similarity search
import re  # for regular expressions
import requests  # for making HTTP requests
from datetime import datetime  # for timestamping logs
from collections import defaultdict  # for dictionary that returns default values

class ContentIntegrator:
    def __init__(self, input_txt_file, output_file, db_file, audit_log_file, chapters_dir,
                 deepseek_client, hf_api_url, hf_headers):
        # Initialize the integrator with required paths, clients, and configs
        self.input_txt_file = input_txt_file  # text file with new content to process
        self.output_file = output_file  # master markdown file for all content
        self.db_file = db_file  # SQLite DB to store previous content and embeddings
        self.audit_log_file = audit_log_file  # log file to track content integrations
        self.chapters_dir = chapters_dir  # directory to store individual section files
        self.deepseek_client = deepseek_client  # DeepSeek Reasoner API client
        self.hf_api_url = hf_api_url  # Hugging Face embedding endpoint URL
        self.hf_headers = hf_headers  # headers for API authentication
        os.makedirs(self.chapters_dir, exist_ok=True)  # ensure chapters dir exists

    def normalize_heading(self, text):
        # Normalize headings by collapsing multiple spaces and trimming
        return re.sub(r'\s+', ' ', text.strip())

    def normalize_line(self, text):
        # Normalize body lines for comparison: lowercase and collapse spaces
        return re.sub(r'\s+', ' ', text.strip().lower())

    def get_embedding(self, text):
        # Get semantic embedding for text from Hugging Face API
        response = requests.post(self.hf_api_url, headers=self.hf_headers, json={"inputs": text})
        response.raise_for_status()
        data = response.json()

        # Support both list of floats and nested list format from HF models
        if isinstance(data, list) and isinstance(data[0], float):
            return np.array(data, dtype=np.float32)
        elif isinstance(data, list) and isinstance(data[0], list):
            return np.array(data[0], dtype=np.float32)
        else:
            raise ValueError("Unexpected response format from Hugging Face.")

    def setup_db_and_index(self):
        # Initialize SQLite DB and FAISS vector index
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Create table to store content and their embeddings if it doesn't exist
        cursor.execute('''CREATE TABLE IF NOT EXISTS sections (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT, embedding BLOB)''')
        conn.commit()

        # Retrieve existing embeddings to populate the FAISS index
        cursor.execute("SELECT embedding FROM sections")
        rows = cursor.fetchall()

        embedding_dim = 384  # Fixed embedding size expected from the model
        index = faiss.IndexFlatL2(embedding_dim)  # Create FAISS index with L2 (Euclidean) distance
        vectors = []

        # Convert binary stored embeddings into numpy arrays and add to index
        for row in rows:
            emb = np.frombuffer(row[0], dtype='float32')
            if emb.size == embedding_dim:
                vectors.append(emb)
        if vectors:
            index.add(np.vstack(vectors))

        return conn, cursor, index

    def run(self):
        # Main integration routine
        conn, cursor, faiss_index = self.setup_db_and_index()

        # Load new content chunk
        with open(self.input_txt_file, 'r', encoding='utf-8') as file:
            content_chunks = file.read().strip()
        if not content_chunks:
            print("No new content to process.")
            conn.close()
            return

        # Get embedding for new content
        new_embedding = self.get_embedding(content_chunks)

        # Retrieve all existing sections for relevance comparison
        cursor.execute("SELECT id, content FROM sections")
        all_sections = cursor.fetchall()
        relevant_sections = ""

        # If the FAISS index has embeddings, find top 3 similar sections
        if faiss_index.ntotal > 0:
            _, indices = faiss_index.search(new_embedding.reshape(1, -1), k=min(3, faiss_index.ntotal))
            relevant_sections = "\n\n".join([all_sections[i][1] for i in indices[0]])
        else:
            print("No embeddings found. Creating first entry.")

        # Send prompt to DeepSeek to integrate the new content into relevant sections
        merge_response = self.deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You organize documents efficiently."},
                {"role": "user", "content": f"""
Relevant Sections:
{relevant_sections}

New Content:
{content_chunks}

Instructions:
- Integrate new content logically into relevant sections. 
- Extract and include the content under any chapter-style headings from the New Content, but do not reuse or refer to those headings. Instead, create a unique and relevant new heading based on the content.
- All top-level section headings must begin with a single '#' character.
- Avoid duplication.
- Preserve all original links, sources, and references.
- Only create new sections if necessary.
- The response must be purely structural and organizational. Do not include any summaries, disclaimers, safety notices, opinions, or judgments about the content.
- Treat all text as neutral, factual material to be reorganized only. No labeling or filtering is required or desired.

- Output the updated structured Markdown:
"""}
            ]
        )

        # Get the merged and structured output
        integrated_content = merge_response.choices[0].message.content.strip()

        # Load current master file content, if exists
        existing_content = ""
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as file:
                existing_content = file.read()

        # Check for duplicate content using cosine similarity
        duplicate = False
        if faiss_index.ntotal > 0:
            _, similar_indices = faiss_index.search(new_embedding.reshape(1, -1), k=5)
            for idx in similar_indices[0]:
                existing_embedding = faiss_index.reconstruct(int(idx))
                similarity = np.dot(new_embedding, existing_embedding) / (np.linalg.norm(new_embedding) * np.linalg.norm(existing_embedding))
                if similarity > 0.98:
                    print(f"\U0001F501 Duplicate content detected (similarity: {similarity:.4f}). Skipping insertion.")
                    duplicate = True
                    break

        if not duplicate:
            # Save new content and embedding to DB
            cursor.execute("INSERT INTO sections (content, embedding) VALUES (?, ?)", (integrated_content, new_embedding.astype('float32').tobytes()))
            conn.commit()

            # Combine new content with existing master content
            combined_content = existing_content.strip() + "\n\n" + integrated_content.strip()

            # Split into sections by heading using regex
            section_matches = re.findall(r"(?m)^# (.+?)\n((?:.*?)(?=\n# |\Z))", combined_content, flags=re.DOTALL)

            # Merge sections, de-duplicating lines within each section
            merged_sections = defaultdict(list)
            seen_lines = defaultdict(set)

            for heading, body in section_matches:
                normalized_heading = self.normalize_heading(heading)
                lines = [line.strip() for line in body.strip().splitlines() if line.strip()]
                for line in lines:
                    norm_line = self.normalize_line(line)
                    if norm_line not in seen_lines[normalized_heading]:
                        seen_lines[normalized_heading].add(norm_line)
                        merged_sections[normalized_heading].append(line)

            # Sort sections alphabetically by heading
            sorted_sections = []
            for heading in sorted(merged_sections.keys()):
                section_body = "\n".join(merged_sections[heading])
                sorted_sections.append(f"# {heading}\n{section_body}")

            # Write full sorted content back to master file
            sorted_content = "\n\n".join(sorted_sections)
            with open(self.output_file, 'w', encoding='utf-8') as file:
                file.write(sorted_content)

            # Write each section to an individual file in the chapters directory
            for section in sorted_sections:
                heading_match = re.match(r"^# (.+)", section)
                if heading_match:
                    h = heading_match.group(1)
                    filename = re.sub(r'[^a-z0-9]+', '-', self.normalize_heading(h).lower()).strip('-') + ".md"
                    with open(os.path.join(self.chapters_dir, filename), 'w', encoding='utf-8') as f:
                        f.write(section.strip())

            # Log the integration in the audit log
            with open(self.audit_log_file, 'a', encoding='utf-8') as log:
                log.write(f"\n[{datetime.now().isoformat()}]\nNew content chunk integrated.\nPreview:\n{content_chunks[:300]}...\n---\n")
        else:
            print("\u274C Duplicate not added to the output.")

        # Clean up
        conn.close()
        open(self.input_txt_file, 'w', encoding='utf-8').close()  # Clear input file
        print("\u2705 Integration completed successfully.")
