"""
Semantic Chunking Module for VLM Document Parsing Evaluation

This module provides:
- Text chunking strategies (fixed, recursive, semantic, hierarchical)
- Label-free quality metrics (BC, CS) based on MoC paper (arXiv:2503.09600v2)
- LLM client for perplexity-based metric calculation
"""

from .chunker import (
    ChunkingStrategy,
    ChunkerConfig,
    Chunk,
    TextChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    HierarchicalChunker,
    FixedSizeChunker,
    create_chunker,
)
from .metrics import (
    # Data classes
    BCScore,
    CSScore,
    ChunkingMetrics,
    # LLM clients
    LLMClient,
    MockLLMClient,
    # Functions
    calculate_bc,
    calculate_cs,
    calculate_edge_weight,
    calculate_structural_entropy,
    build_chunk_graph,
    evaluate_chunking,
    compare_chunking_quality,
)

__all__ = [
    # Chunker classes
    "ChunkingStrategy",
    "ChunkerConfig",
    "Chunk",
    "TextChunker",
    "RecursiveCharacterChunker",
    "SemanticChunker",
    "HierarchicalChunker",
    "FixedSizeChunker",
    "create_chunker",
    # Metrics - Data classes
    "BCScore",
    "CSScore",
    "ChunkingMetrics",
    # Metrics - LLM clients
    "LLMClient",
    "MockLLMClient",
    # Metrics - Functions
    "calculate_bc",
    "calculate_cs",
    "calculate_edge_weight",
    "calculate_structural_entropy",
    "build_chunk_graph",
    "evaluate_chunking",
    "compare_chunking_quality",
]
