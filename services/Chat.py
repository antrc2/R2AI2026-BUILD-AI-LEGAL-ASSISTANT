import os
from typing import List, Dict, Any, Generator, Optional
from dotenv import load_dotenv
from openai import OpenAI
from services.OpenAIExtended import OpenAIExtended

load_dotenv()

# --- CẤU HÌNH CHAT ---
CHAT_MODEL_NAME = os.getenv('CHAT_MODEL_NAME', 'qwen3-4b')
CHAT_BASE_URL = os.getenv("CHAT_BASE_URL", "http://localhost:1234/v1")
CHAT_API_KEY = os.getenv("CHAT_API_KEY", 'dont need')

# --- CẤU HÌNH EMBEDDING ---
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'text-embedding-qwen3-embedding-0.6b')
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:1234/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", 'dont need')

# --- CẤU HÌNH RERANK ---
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME")
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL")
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY")


class ChatService:
    def __init__(self):
        # Client cho Chat (LLM)
        self.chat_client = OpenAI(
            base_url=CHAT_BASE_URL,
            api_key=CHAT_API_KEY
        )
        
        # Client cho Embedding
        self.embedding_client = OpenAI(
            base_url=EMBEDDING_BASE_URL,
            api_key=EMBEDDING_API_KEY
        )
        
        # Client cho Rerank (optional)
        if (
            RERANKER_MODEL_NAME
            and RERANKER_BASE_URL
            and RERANKER_API_KEY
        ):
            self.rerank_client = OpenAIExtended(
                base_url=RERANKER_BASE_URL,
                api_key=RERANKER_API_KEY
            )
        else:
            self.rerank_client = None

    def get_embedding(self, text: str) -> List[float]:
        """Lấy vector embedding từ model."""
        try:
            response = self.embedding_client.embeddings.create(
                model=EMBEDDING_MODEL_NAME,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Lỗi embedding: {e}")
            return []

    def get_rerank_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Nếu không cấu hình reranker thì trả về score mặc định
        để giữ nguyên thứ tự documents.
        """
        if not documents:
            return []

        # Không dùng reranker -> giữ nguyên thứ tự
        if self.rerank_client is None:
            return [1.0] * len(documents)

        try:
            response = self.rerank_client.reranker.create(
                model=RERANKER_MODEL_NAME,
                query=query,
                documents=documents
            )

            score_map = {
                res.index: res.relevance_score
                for res in response.results
            }

            return [
                score_map.get(i, 0.0)
                for i in range(len(documents))
            ]

        except Exception as e:
            print(f"Lỗi rerank: {e}")
            return [1.0] * len(documents)
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        response_format: Optional[Dict[str, str]] = None,
        stream: bool = False
    ) -> Generator[Any, None, None]:
        """Gọi LLM chat completion."""
        kwargs = {
            "model": CHAT_MODEL_NAME,
            "messages": messages,
            "stream": stream,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = self.chat_client.chat.completions.create(**kwargs)

            if stream:
                for chunk in response:
                    yield chunk
            else:
                yield response.choices[0].message
        except Exception as e:
            error_msg = f"Lỗi khi gọi LLM: {str(e)}"
            print(error_msg)
            if stream:
                yield {"error": error_msg}
            else:
                raise Exception(error_msg)