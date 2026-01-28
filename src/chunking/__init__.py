"""
Semantic Chunking Module for VLM Document Parsing Evaluation

This module provides:
- Semantic text chunking using LangChain SemanticChunker
- Label-free quality metrics (BC, CS) based on MoC paper (arXiv:2503.09600v2)
- Embedding-based semantic distance calculation for BC/CS metrics
"""

from .chunker import (
    ChunkerConfig,
    Chunk,
    SemanticChunker,
    create_chunker,
)
from .embeddings import APIEmbeddings
from .metrics import (
    # Data classes
    BCScore,
    CSScore,
    ChunkingMetrics,
    # Embedding clients
    EmbeddingClient,
    MockEmbeddingClient,
    create_embedding_client,
    # Functions
    calculate_bc,
    calculate_cs,
    calculate_edge_weight_semantic,
    calculate_structural_entropy,
    build_chunk_graph,
    evaluate_chunking,
    compare_chunking_quality,
)
from .dashboard_export import (
    SentenceBCData,
    StrategyResult,
    split_sentences,
    calculate_bc_by_sentence,
    calculate_cs_by_chunk,
    export_parser_chunking_results,
)

__all__ = [
    # Chunker classes
    "ChunkerConfig",
    "Chunk",
    "SemanticChunker",
    "create_chunker",
    # Embeddings
    "APIEmbeddings",
    # Metrics - Data classes
    "BCScore",
    "CSScore",
    "ChunkingMetrics",
    # Metrics - Embedding clients
    "EmbeddingClient",
    "MockEmbeddingClient",
    "create_embedding_client",
    # Metrics - Functions
    "calculate_bc",
    "calculate_cs",
    "calculate_edge_weight_semantic",
    "calculate_structural_entropy",
    "build_chunk_graph",
    "evaluate_chunking",
    "compare_chunking_quality",
    # Dashboard export
    "SentenceBCData",
    "StrategyResult",
    "split_sentences",
    "calculate_bc_by_sentence",
    "calculate_cs_by_chunk",
    "export_parser_chunking_results",
]
