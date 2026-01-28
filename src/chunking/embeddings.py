"""
LangChain compatible embeddings using Infinity API.

Provides a LangChain Embeddings implementation that calls external embedding APIs.
"""

from typing import List

import httpx
from langchain_core.embeddings import Embeddings


class APIEmbeddings(Embeddings):
    """LangChain compatible embeddings using external API."""

    def __init__(
        self,
        api_url: str = "http://localhost:8001/embeddings",
        model: str = "BAAI/bge-m3",
        timeout: float = 60.0,
    ):
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        response = httpx.post(
            self.api_url,
            json={"model": self.model, "input": texts},
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]
