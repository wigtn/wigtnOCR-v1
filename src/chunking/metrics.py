"""
MoC-based Chunking Quality Metrics

This module implements label-free chunking evaluation metrics based on the
MoC (Mixtures of Chunking) paper (arXiv:2503.09600v2).

Key Metrics:
- BC (Boundary Clarity): Measures independence between adjacent chunks
- CS (Chunk Stickiness): Measures overall graph connectivity via Structural Entropy

Advantages over traditional metrics:
- No Ground Truth required
- Repeatable measurements in production
- Strong correlation with RAG performance (BC↔ROUGE-L: 0.88)
"""

import math
import httpx
from dataclasses import dataclass
from typing import Optional
from collections.abc import Sequence


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BCScore:
    """Boundary Clarity evaluation result.

    BC = ppl(q|d) / ppl(q)
    - Higher is better (chunks are more independent)
    - Close to 1.0: chunks are independent
    - Close to 0.0: chunks are highly dependent
    """
    score: float  # Average BC across all adjacent pairs
    pair_scores: list[float]  # Per-pair BC scores
    min_score: float
    max_score: float
    std_dev: float
    pair_count: int

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "pair_count": self.pair_count,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "std_dev": self.std_dev,
        }


@dataclass
class CSScore:
    """Chunk Stickiness evaluation result.

    CS = -Σ (h_i / 2m) * log2(h_i / 2m)  (Structural Entropy)
    - Lower is better (chunks are more independent)
    - h_i: degree of node i
    - m: total number of edges
    """
    score: float  # Structural Entropy
    graph_type: str  # "complete" or "incomplete"
    node_count: int
    edge_count: int
    threshold_k: float

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "graph_type": self.graph_type,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "threshold_k": self.threshold_k,
        }


@dataclass
class ChunkingMetrics:
    """Combined chunking quality metrics (BC + CS)."""
    bc_score: Optional[BCScore] = None
    cs_score: Optional[CSScore] = None

    def to_dict(self) -> dict:
        return {
            "bc": self.bc_score.to_dict() if self.bc_score else None,
            "cs": self.cs_score.to_dict() if self.cs_score else None,
        }


# =============================================================================
# LLM Client for Perplexity Calculation
# =============================================================================

class LLMClient:
    """Client for LLM API with perplexity calculation support.

    Supports OpenAI-compatible APIs (vLLM, text-generation-inference, etc.)
    that can return log probabilities.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1/completions",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: float = 60.0,
        api_key: str = "dummy"  # For local APIs
    ):
        """Initialize LLM client.

        Args:
            api_url: API endpoint URL (OpenAI-compatible)
            model: Model ID
            timeout: Request timeout in seconds
            api_key: API key (use "dummy" for local APIs)
        """
        self.api_url = api_url
        self.model = model
        self.timeout = timeout
        self.api_key = api_key

    def calculate_perplexity(
        self,
        text: str,
        context: Optional[str] = None
    ) -> float:
        """Calculate perplexity of text, optionally conditioned on context.

        Args:
            text: Target text to calculate perplexity for
            context: Optional context to condition on

        Returns:
            Perplexity value (lower = more predictable)
        """
        if not text.strip():
            return 1.0

        # Build prompt
        if context:
            prompt = f"{context}\n\n{text}"
            # We want PPL of text given context
            _ = context + "\n\n"  # echo_prompt for future use
        else:
            prompt = text
            _ = ""  # echo_prompt placeholder for future use

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.api_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "max_tokens": 1,  # We only need logprobs, not generation
                        "logprobs": True,
                        "echo": True,  # Return logprobs for input tokens too
                    }
                )
                response.raise_for_status()
                result = response.json()

                # Extract log probabilities
                logprobs = result["choices"][0].get("logprobs", {})
                token_logprobs = logprobs.get("token_logprobs", [])

                if not token_logprobs:
                    return 1.0

                # Filter out None values (usually the first token)
                valid_logprobs = [lp for lp in token_logprobs if lp is not None]

                if not valid_logprobs:
                    return 1.0

                # Calculate perplexity: exp(-mean(log_probs))
                avg_neg_log_prob = -sum(valid_logprobs) / len(valid_logprobs)
                perplexity = math.exp(avg_neg_log_prob)

                return perplexity

        except Exception as e:
            print(f"Warning: Perplexity calculation failed: {e}")
            return 1.0

    def calculate_perplexity_batch(
        self,
        texts: list[str],
        contexts: Optional[list[Optional[str]]] = None
    ) -> list[float]:
        """Calculate perplexity for multiple texts.

        Args:
            texts: List of target texts
            contexts: Optional list of contexts (same length as texts)

        Returns:
            List of perplexity values
        """
        if contexts is None:
            contexts = [None] * len(texts)

        return [
            self.calculate_perplexity(text, context)
            for text, context in zip(texts, contexts)
        ]


# =============================================================================
# Mock LLM Client (for testing without API)
# =============================================================================

class MockLLMClient:
    """Mock LLM client for testing without actual API.

    Uses simple heuristics based on text length and overlap.
    """

    def __init__(self):
        pass

    def calculate_perplexity(
        self,
        text: str,
        context: Optional[str] = None
    ) -> float:
        """Mock perplexity calculation using text statistics.

        Heuristic:
        - Base perplexity from vocabulary diversity
        - Context reduces perplexity if texts share words
        """
        if not text.strip():
            return 1.0

        # Simple vocabulary-based heuristic
        words = text.lower().split()
        unique_words = set(words)
        vocab_diversity = len(unique_words) / max(len(words), 1)

        # Base perplexity (higher diversity = higher perplexity)
        base_ppl = 10 + vocab_diversity * 90  # Range: 10-100

        if context:
            # Reduce perplexity based on word overlap
            context_words = set(context.lower().split())
            overlap = len(unique_words & context_words)
            overlap_ratio = overlap / max(len(unique_words), 1)

            # More overlap = lower conditional perplexity
            base_ppl *= (1 - overlap_ratio * 0.5)

        return max(base_ppl, 1.0)

    def calculate_perplexity_batch(
        self,
        texts: list[str],
        contexts: Optional[list[Optional[str]]] = None
    ) -> list[float]:
        if contexts is None:
            contexts = [None] * len(texts)
        return [
            self.calculate_perplexity(text, context)
            for text, context in zip(texts, contexts)
        ]


# =============================================================================
# BC (Boundary Clarity) Calculation
# =============================================================================

def calculate_bc(
    chunks: Sequence,
    llm_client: LLMClient | MockLLMClient,
    verbose: bool = False
) -> BCScore:
    """Calculate Boundary Clarity for a list of chunks.

    BC measures how independent adjacent chunks are.
    BC(q, d) = ppl(q|d) / ppl(q)

    Args:
        chunks: List of Chunk objects or strings
        llm_client: LLM client for perplexity calculation
        verbose: Print progress

    Returns:
        BCScore with average and per-pair scores
    """
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]

    if len(contents) < 2:
        return BCScore(
            score=1.0,
            pair_scores=[],
            min_score=1.0,
            max_score=1.0,
            std_dev=0.0,
            pair_count=0,
        )

    pair_scores = []

    for i in range(len(contents) - 1):
        d = contents[i]      # Current chunk (context)
        q = contents[i + 1]  # Next chunk (target)

        if verbose:
            print(f"  BC calculation: chunk {i} → {i+1}", end="", flush=True)

        # Calculate perplexities
        ppl_q = llm_client.calculate_perplexity(q)
        ppl_q_given_d = llm_client.calculate_perplexity(q, context=d)

        # BC = ppl(q|d) / ppl(q)
        # Higher BC = more independent (good)
        if ppl_q > 0:
            bc = ppl_q_given_d / ppl_q
        else:
            bc = 1.0

        # Clamp to reasonable range
        bc = max(0.0, min(bc, 2.0))
        pair_scores.append(bc)

        if verbose:
            print(f" → BC={bc:.4f} (ppl_q={ppl_q:.2f}, ppl_q|d={ppl_q_given_d:.2f})")

    if not pair_scores:
        return BCScore(
            score=1.0,
            pair_scores=[],
            min_score=1.0,
            max_score=1.0,
            std_dev=0.0,
            pair_count=0,
        )

    import numpy as np
    return BCScore(
        score=float(np.mean(pair_scores)),
        pair_scores=pair_scores,
        min_score=float(min(pair_scores)),
        max_score=float(max(pair_scores)),
        std_dev=float(np.std(pair_scores)),
        pair_count=len(pair_scores),
    )


# =============================================================================
# CS (Chunk Stickiness) Calculation
# =============================================================================

def calculate_edge_weight(
    q: str,
    d: str,
    llm_client: LLMClient | MockLLMClient
) -> float:
    """Calculate edge weight between two chunks.

    Edge(q, d) = (ppl(q) - ppl(q|d)) / ppl(q)
    - Close to 1: high correlation (d helps predict q)
    - Close to 0: independent

    Args:
        q: Target chunk
        d: Context chunk
        llm_client: LLM client

    Returns:
        Edge weight [0, 1]
    """
    ppl_q = llm_client.calculate_perplexity(q)
    ppl_q_given_d = llm_client.calculate_perplexity(q, context=d)

    if ppl_q <= 0:
        return 0.0

    weight = (ppl_q - ppl_q_given_d) / ppl_q
    return max(0.0, min(1.0, weight))


def build_chunk_graph(
    chunks: Sequence,
    llm_client: LLMClient | MockLLMClient,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    verbose: bool = False
) -> dict[int, list[tuple[int, float]]]:
    """Build a weighted graph from chunks based on edge weights.

    Args:
        chunks: List of chunks
        llm_client: LLM client for perplexity
        threshold_k: Only keep edges with weight >= threshold_k
        graph_type: "complete" (all pairs) or "incomplete" (sequential only)
        verbose: Print progress

    Returns:
        Adjacency list: {node_id: [(neighbor_id, weight), ...]}
    """
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]
    n = len(contents)

    if n == 0:
        return {}

    # Initialize adjacency list
    graph: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

    if graph_type == "complete":
        # Complete graph: all pairs
        total_pairs = n * (n - 1) // 2
        current = 0

        for i in range(n):
            for j in range(i + 1, n):
                current += 1
                if verbose:
                    print(f"  Edge {current}/{total_pairs}: {i} ↔ {j}", end="", flush=True)

                # Calculate bidirectional weights
                w_ij = calculate_edge_weight(contents[j], contents[i], llm_client)
                w_ji = calculate_edge_weight(contents[i], contents[j], llm_client)

                # Use max weight for undirected graph
                weight = max(w_ij, w_ji)

                if verbose:
                    print(f" → w={weight:.4f}")

                if weight >= threshold_k:
                    graph[i].append((j, weight))
                    graph[j].append((i, weight))

    elif graph_type == "incomplete":
        # Incomplete graph: only sequential pairs (j - i > 0)
        for i in range(n - 1):
            j = i + 1
            if verbose:
                print(f"  Edge: {i} → {j}", end="", flush=True)

            weight = calculate_edge_weight(contents[j], contents[i], llm_client)

            if verbose:
                print(f" → w={weight:.4f}")

            if weight >= threshold_k:
                graph[i].append((j, weight))
                graph[j].append((i, weight))

    return graph


def calculate_structural_entropy(graph: dict[int, list[tuple[int, float]]]) -> float:
    """Calculate Structural Entropy of the graph.

    H = -Σ (h_i / 2m) * log2(h_i / 2m)

    Where:
    - h_i: weighted degree of node i
    - m: total edge weight / 2

    Args:
        graph: Adjacency list with weights

    Returns:
        Structural entropy (lower = better chunking)
    """
    if not graph:
        return 0.0

    # Calculate weighted degrees
    degrees = {}
    for node, neighbors in graph.items():
        degrees[node] = sum(w for _, w in neighbors)

    # Total edge weight (divide by 2 for undirected)
    total_weight = sum(degrees.values())
    m = total_weight / 2

    if m <= 0:
        return 0.0

    # Calculate structural entropy
    entropy = 0.0
    for node, h_i in degrees.items():
        if h_i > 0:
            p = h_i / (2 * m)
            entropy -= p * math.log2(p)

    return entropy


def calculate_cs(
    chunks: Sequence,
    llm_client: LLMClient | MockLLMClient,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    verbose: bool = False
) -> CSScore:
    """Calculate Chunk Stickiness using Structural Entropy.

    Args:
        chunks: List of chunks
        llm_client: LLM client for perplexity
        threshold_k: Edge filtering threshold
        graph_type: "complete" or "incomplete"
        verbose: Print progress

    Returns:
        CSScore with structural entropy
    """
    contents = [
        c.content if hasattr(c, 'content') else str(c)
        for c in chunks
    ]

    if len(contents) < 2:
        return CSScore(
            score=0.0,
            graph_type=graph_type,
            node_count=len(contents),
            edge_count=0,
            threshold_k=threshold_k,
        )

    # Build graph
    if verbose:
        print(f"  Building {graph_type} graph (threshold={threshold_k})...")

    graph = build_chunk_graph(
        chunks, llm_client, threshold_k, graph_type, verbose
    )

    # Count edges (divide by 2 for undirected)
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2

    # Calculate structural entropy
    entropy = calculate_structural_entropy(graph)

    return CSScore(
        score=entropy,
        graph_type=graph_type,
        node_count=len(contents),
        edge_count=edge_count,
        threshold_k=threshold_k,
    )


# =============================================================================
# Combined Evaluation
# =============================================================================

def evaluate_chunking(
    chunks: Sequence,
    llm_client: LLMClient | MockLLMClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    calculate_cs_flag: bool = True,
    verbose: bool = False
) -> ChunkingMetrics:
    """Evaluate chunking quality using BC and CS metrics.

    Args:
        chunks: List of chunks
        llm_client: LLM client (uses MockLLMClient if None)
        threshold_k: CS edge filtering threshold
        graph_type: CS graph type
        calculate_cs_flag: Whether to calculate CS (can be slow)
        verbose: Print progress

    Returns:
        ChunkingMetrics with BC and CS scores
    """
    if llm_client is None:
        print("Warning: No LLM client provided, using MockLLMClient")
        llm_client = MockLLMClient()

    # Calculate BC
    if verbose:
        print("Calculating BC (Boundary Clarity)...")
    bc_score = calculate_bc(chunks, llm_client, verbose)

    # Calculate CS
    cs_score = None
    if calculate_cs_flag:
        if verbose:
            print("Calculating CS (Chunk Stickiness)...")
        cs_score = calculate_cs(chunks, llm_client, threshold_k, graph_type, verbose)

    return ChunkingMetrics(bc_score=bc_score, cs_score=cs_score)


def compare_chunking_quality(
    results: dict[str, Sequence],
    llm_client: LLMClient | MockLLMClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    verbose: bool = False
) -> dict:
    """Compare chunking quality across multiple parsers.

    Args:
        results: {parser_name: chunks} dictionary
        llm_client: LLM client
        threshold_k: CS threshold
        graph_type: CS graph type
        verbose: Print progress

    Returns:
        Comparison results with metrics per parser
    """
    comparison = {}

    for parser_name, chunks in results.items():
        if verbose:
            print(f"\n=== Evaluating: {parser_name} ===")

        metrics = evaluate_chunking(
            chunks, llm_client, threshold_k, graph_type,
            calculate_cs_flag=True, verbose=verbose
        )

        comparison[parser_name] = {
            "metrics": metrics.to_dict(),
            "chunk_count": len(chunks),
        }

    return comparison
