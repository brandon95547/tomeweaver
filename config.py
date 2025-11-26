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
        self.similarity_threshold = 0.90
        self.dimensions = 384

    @classmethod
    def from_env(cls):
        return cls()
