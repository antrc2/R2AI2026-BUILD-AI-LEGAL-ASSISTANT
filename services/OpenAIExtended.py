import requests
from typing import List, Optional
from openai import OpenAI
from pydantic import BaseModel


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class Document(BaseModel):
    text: str
    multi_modal: Optional[dict] = None


class RerankerResult(BaseModel):
    index: int
    document: Document
    relevance_score: float


class RerankerResponse(BaseModel):
    id: str
    model: str
    usage: Usage
    results: List[RerankerResult]


class Reranker:
    def __init__(self, client: OpenAI):
        self._client = client

    def create(
        self,
        model: str,
        query: str,
        documents: List[str],
    ) -> RerankerResponse:
        """
        Gọi API rerank với tham số query và documents riêng biệt.
        """
        response = requests.post(
            f"{self._client.base_url}rerank",
            headers={
                "Authorization": f"Bearer {self._client.api_key}"
            },
            json={
                "model": model,
                "query": query,
                "documents": documents,
            },
        )

        response.raise_for_status()
        return RerankerResponse.model_validate(response.json())


class OpenAIExtended(OpenAI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = Reranker(self)