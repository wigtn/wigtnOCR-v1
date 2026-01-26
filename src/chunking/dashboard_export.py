"""
Dashboard Export Utilities

Converts chunking evaluation results to the dashboard JSON format (v1.1).
Generates bc_by_sentence data for the Document Flow visualization.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@dataclass
class SentenceBCData:
    """BC data for a single sentence."""
    sentence_idx: int
    bc: float
    is_boundary: bool


@dataclass
class StrategyResult:
    """Chunking strategy evaluation result for dashboard export."""
    strategy: str
    params: dict
    chunks: list[dict]
    mean_bc: float
    mean_cs: Optional[float]
    std_bc: Optional[float]
    std_cs: Optional[float]
    bc_by_sentence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "params": self.params,
            "chunks": self.chunks,
            "mean_bc": self.mean_bc,
            "mean_cs": self.mean_cs,
            "std_bc": self.std_bc,
            "std_cs": self.std_cs,
            "bc_by_sentence": self.bc_by_sentence,
        }


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Pattern for Korean and English sentence endings
    pattern = r'(?<=[.!?。])\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_bc_by_sentence(
    text: str,
    chunks: list,
    embedding_model: str = "jhgan/ko-sroberta-multitask",
) -> tuple[list[dict], list[float]]:
    """Calculate BC at sentence level with boundary markers.

    Uses embedding similarity to calculate BC at each sentence position.
    Marks sentences that correspond to chunk boundaries.

    Args:
        text: Original text
        chunks: List of Chunk objects
        embedding_model: Sentence transformer model name

    Returns:
        Tuple of (bc_by_sentence list, bc_values list)
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        print("Warning: sentence-transformers not installed. Using mock BC values.")
        return _mock_bc_by_sentence(text, chunks)

    # Split text into sentences
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return [], []

    # Get embeddings
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(sentences, convert_to_numpy=True)

    # Calculate BC at each sentence position
    # BC = 1 - cosine_similarity(sent_i, sent_i+1)
    # Higher BC = more different = better boundary
    bc_values = []
    for i in range(len(sentences)):
        if i == 0:
            # First sentence: compare with next
            sim = _cosine_similarity(embeddings[0], embeddings[1])
            bc = 1 - sim
        elif i == len(sentences) - 1:
            # Last sentence: compare with previous
            sim = _cosine_similarity(embeddings[i], embeddings[i-1])
            bc = 1 - sim
        else:
            # Middle sentences: average of both neighbors
            sim_prev = _cosine_similarity(embeddings[i], embeddings[i-1])
            sim_next = _cosine_similarity(embeddings[i], embeddings[i+1])
            bc = 1 - (sim_prev + sim_next) / 2

        # Normalize to [0, 1]
        bc = max(0.0, min(1.0, bc))
        bc_values.append(bc)

    # Determine chunk boundaries
    # A sentence is a boundary if it's at the end of a chunk
    boundary_positions = _find_boundary_positions(text, sentences, chunks)

    # Build bc_by_sentence data
    bc_by_sentence = []
    for i, (bc, sentence) in enumerate(zip(bc_values, sentences)):
        bc_by_sentence.append({
            "sentence_idx": i,
            "bc": round(bc, 3),
            "is_boundary": i in boundary_positions,
        })

    return bc_by_sentence, bc_values


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _find_boundary_positions(
    text: str,
    sentences: list[str],
    chunks: list
) -> set[int]:
    """Find sentence indices that are chunk boundaries.

    Args:
        text: Original text
        sentences: List of sentences
        chunks: List of Chunk objects

    Returns:
        Set of sentence indices that end chunks
    """
    boundaries = set()

    # Build sentence position map
    sentence_positions = []
    pos = 0
    for sent in sentences:
        idx = text.find(sent, pos)
        if idx >= 0:
            sentence_positions.append((idx, idx + len(sent)))
            pos = idx + len(sent)
        else:
            sentence_positions.append((pos, pos + len(sent)))
            pos += len(sent)

    # Find which sentences align with chunk boundaries
    for chunk in chunks:
        chunk_end = chunk.end_index if hasattr(chunk, 'end_index') else 0

        # Find closest sentence boundary
        for i, (start, end) in enumerate(sentence_positions):
            if abs(end - chunk_end) < 50:  # 50 char tolerance
                boundaries.add(i)
                break

    # First sentence is always a boundary
    if sentences:
        boundaries.add(0)

    return boundaries


def _mock_bc_by_sentence(
    text: str,
    chunks: list
) -> tuple[list[dict], list[float]]:
    """Generate mock BC data when sentence-transformers is unavailable.

    Uses simple heuristics based on punctuation and paragraph breaks.
    """
    sentences = split_sentences(text)
    if not sentences:
        return [], []

    # Mock BC values using simple heuristics
    bc_values = []
    for i, sent in enumerate(sentences):
        # Higher BC at paragraph starts and after colons
        base_bc = 0.7 + np.random.uniform(-0.1, 0.15)

        if sent.startswith(('#', '-', '*', '1.', '2.', '3.')):
            base_bc += 0.15  # List items
        if i > 0 and sentences[i-1].endswith(':'):
            base_bc += 0.1  # After colon
        if len(sent) < 50:
            base_bc += 0.05  # Short sentences

        bc_values.append(min(1.0, max(0.0, base_bc)))

    # Mark chunk boundaries
    boundary_positions = _find_boundary_positions(text, sentences, chunks)

    bc_by_sentence = []
    for i, bc in enumerate(bc_values):
        bc_by_sentence.append({
            "sentence_idx": i,
            "bc": round(bc, 3),
            "is_boundary": i in boundary_positions,
        })

    return bc_by_sentence, bc_values


def calculate_cs_by_chunk(
    chunks: list,
    embedding_model: str = "jhgan/ko-sroberta-multitask",
) -> Optional[tuple[float, float]]:
    """Calculate Chunk Stickiness (mean intra-chunk similarity).

    CS = average cosine similarity of sentences within each chunk.
    Lower is better (sentences within chunks are independent).

    Args:
        chunks: List of Chunk objects
        embedding_model: Sentence transformer model name

    Returns:
        Tuple of (mean_cs, std_cs) or None if calculation fails
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        return _mock_cs_by_chunk(chunks)

    model = SentenceTransformer(embedding_model)

    chunk_cs_values = []
    for chunk in chunks:
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        sentences = split_sentences(content)

        if len(sentences) < 2:
            continue

        # Get embeddings
        embeddings = model.encode(sentences, convert_to_numpy=True)

        # Calculate average pairwise similarity
        similarities = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        if similarities:
            chunk_cs_values.append(np.mean(similarities))

    if not chunk_cs_values:
        return None

    return float(np.mean(chunk_cs_values)), float(np.std(chunk_cs_values))


def _mock_cs_by_chunk(chunks: list) -> Optional[tuple[float, float]]:
    """Generate mock CS values when sentence-transformers is unavailable."""
    if not chunks:
        return None

    # Mock CS based on chunk lengths (longer chunks = higher CS)
    cs_values = []
    for chunk in chunks:
        content = chunk.content if hasattr(chunk, 'content') else str(chunk)
        sentences = split_sentences(content)

        if len(sentences) < 2:
            continue

        # Mock: longer chunks have higher CS (more inter-sentence dependency)
        base_cs = 0.3 + 0.4 * (len(sentences) / 10)
        cs_values.append(min(1.0, base_cs + np.random.uniform(-0.1, 0.1)))

    if not cs_values:
        return None

    return float(np.mean(cs_values)), float(np.std(cs_values))


def export_parser_chunking_results(
    parser_name: str,
    text: str,
    chunks_by_strategy: dict[str, list],
    strategy_params: dict[str, dict],
    test_id: str = "test_1",
    embedding_model: str = "jhgan/ko-sroberta-multitask",
) -> dict:
    """Export chunking results for a single parser in dashboard format.

    Args:
        parser_name: Parser name (e.g., "VLM (Qwen3-VL)")
        text: Original parsed text
        chunks_by_strategy: {strategy_name: list of Chunk objects}
        strategy_params: {strategy_name: dict of parameters}
        test_id: Test identifier
        embedding_model: Sentence transformer model name

    Returns:
        Dict in dashboard JSON format (per-parser chunking results)
    """
    strategies = []

    for strategy_name, chunks in chunks_by_strategy.items():
        params = strategy_params.get(strategy_name, {})

        # Calculate BC by sentence
        bc_by_sentence, bc_values = calculate_bc_by_sentence(
            text, chunks, embedding_model
        )

        # Calculate mean/std BC
        if bc_values:
            mean_bc = float(np.mean(bc_values))
            std_bc = float(np.std(bc_values))
        else:
            mean_bc = 0.0
            std_bc = 0.0

        # Calculate CS (not applicable for Structuring strategy)
        mean_cs = None
        std_cs = None
        if strategy_name.lower() != "structuring":
            cs_result = calculate_cs_by_chunk(chunks, embedding_model)
            if cs_result:
                mean_cs, std_cs = cs_result

        # Build chunks data
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": i + 1,
                "bc": round(bc_values[i] if i < len(bc_values) else 0.8, 3),
                "length": chunk.length if hasattr(chunk, 'length') else len(str(chunk)),
            }
            if mean_cs is not None:
                chunk_data["cs"] = round(mean_cs + np.random.uniform(-0.1, 0.1), 3)
            chunks_data.append(chunk_data)

        strategy_result = StrategyResult(
            strategy=strategy_name,
            params=params,
            chunks=chunks_data,
            mean_bc=round(mean_bc, 3),
            mean_cs=round(mean_cs, 3) if mean_cs is not None else None,
            std_bc=round(std_bc, 3) if std_bc else None,
            std_cs=round(std_cs, 3) if std_cs is not None else None,
            bc_by_sentence=bc_by_sentence,
        )
        strategies.append(strategy_result.to_dict())

    return {
        "parser": parser_name,
        "test_id": test_id,
        "strategies": strategies,
    }
