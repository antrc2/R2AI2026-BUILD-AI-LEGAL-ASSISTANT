from .OpenAIExtended import OpenAIExtended
import os
from dotenv import load_dotenv
load_dotenv()

CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME','Qwen3-2B-Q8_0.gguf')
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL","http://localhost:1234/v1")
CHAT_API_KEY = os.getenv("CHAT_API_KEY",'dont need')

EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME','Qwen3-Embedding-0.6B-Q8_0.gguf')
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL","http://localhost:1234/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY",'dont need')

RERANKER_MODEL_NAME = os.getenv('RERANKER_MODEL_NAME','Qwen3-Reranker-0.6B')
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL","http://localhost:11112/v1")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY",'dont need')

chat_client = OpenAIExtended(base_url=CHAT_BASE_URL,api_key=CHAT_BASE_URL)
embedding_client = OpenAIExtended(base_url=EMBEDDING_BASE_URL,api_key=EMBEDDING_API_KEY)
reranker_client = OpenAIExtended(base_url=RERANKER_BASE_URL, api_key = RERANKER_API_KEY )

class Chat:
    def __init__(self):
        pass
    def completions(self,messages):
        pass