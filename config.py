import os
from dotenv import load_dotenv
import openai

class Config:
    def __init__(self):
        load_dotenv()
        self.deepseek_client = openai.OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_endpoint = (
            "https://api-inference.huggingface.co/pipeline/feature-extraction/"
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.similarity_threshold = 0.90
        self.dimensions = 384

    @classmethod
    def from_env(cls):
        return cls()
