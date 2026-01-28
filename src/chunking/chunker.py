"""
Semantic Text Chunking using LangChain

This module provides semantic chunking using LangChain's SemanticChunker.
Uses embedding similarity to detect semantic boundaries.
"""

from dataclasses import dataclass, field
from typing import Optional

from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker


@dataclass
class Chunk:
    """Represents a single text chunk."""
    id: str
    content: str
    start_index: int
    end_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Length of chunk content."""
        return len(self.content)

    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "length": self.length,
            "metadata": self.metadata,
        }


@dataclass
class ChunkerConfig:
    """Configuration for semantic chunker."""
    breakpoint_threshold_type: str = "percentile"  # percentile, standard_deviation, interquartile, gradient
    breakpoint_threshold_amount: float = 95.0
    min_chunk_size: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
            "min_chunk_size": self.min_chunk_size,
        }


class SemanticChunker:
    """Semantic chunker using LangChain's SemanticChunker with API embeddings."""

    def __init__(
        self,
        config: ChunkerConfig,
        embedding_api_url: str = "http://localhost:8001/embeddings",
        embedding_model: str = "BAAI/bge-m3",
    ):
        self.config = config
        self.embedding_api_url = embedding_api_url
        self.embedding_model = embedding_model
        self._chunker = None

    def _get_chunker(self) -> LangChainSemanticChunker:
        """Lazy load LangChain SemanticChunker."""
        if self._chunker is None:
            from src.chunking.embeddings import APIEmbeddings

            embeddings = APIEmbeddings(
                api_url=self.embedding_api_url,
                model=self.embedding_model
            )
            self._chunker = LangChainSemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.config.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.config.breakpoint_threshold_amount,
                min_chunk_size=self.config.min_chunk_size,
                add_start_index=True,
            )
        return self._chunker

    def chunk(self, text: str, document_id: str = "doc") -> list[Chunk]:
        """Split text into semantic chunks."""
        if not text.strip():
            return []

        chunker = self._get_chunker()
        docs = chunker.create_documents([text])

        chunks = []
        for i, doc in enumerate(docs):
            start_index = doc.metadata.get("start_index", 0)
            chunk = Chunk(
                id=f"{document_id}_chunk_{i}",
                content=doc.page_content,
                start_index=start_index,
                end_index=start_index + len(doc.page_content),
                metadata={
                    "document_id": document_id,
                    "chunk_index": i,
                    **doc.metadata
                }
            )
            chunks.append(chunk)

        return chunks


def create_chunker(
    config: Optional[ChunkerConfig] = None,
    embedding_api_url: str = "http://localhost:8001/embeddings",
    embedding_model: str = "BAAI/bge-m3",
) -> SemanticChunker:
    """Factory function to create SemanticChunker."""
    if config is None:
        config = ChunkerConfig()
    return SemanticChunker(config, embedding_api_url, embedding_model)
