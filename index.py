import os
from dotenv import load_dotenv
from openai import OpenAI
from content_integrator import ContentIntegrator

# Load environment variables from .env file
load_dotenv()

# File and directory paths
input_txt_file = 'content_chunks.txt'
output_file = 'structured_output.md'
db_file = 'embeddings.db'
audit_log_file = 'audit_log.txt'
chapters_dir = 'chapters'
os.makedirs(chapters_dir, exist_ok=True)

# Initialize DeepSeek API client
deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

# Initialize and run integrator
integrator = ContentIntegrator(
    input_txt_file=input_txt_file,
    output_file=output_file,
    db_file=db_file,
    audit_log_file=audit_log_file,
    chapters_dir=chapters_dir,
    deepseek_client=deepseek_client,
    hf_api_url="https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2",
    hf_headers={"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
)

integrator.run()