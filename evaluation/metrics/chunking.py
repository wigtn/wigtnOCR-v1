"""Chunking quality metrics: Boundary Clarity (BC) and Chunk Stickiness (CS).

Implements the MoC (Metrics of Chunks) framework from:
    Zhao et al., "Are Long-Context LLMs Better than RAG? A Comprehensive
    Study of LLMs' Ability to Utilize Evidence in Long-Context vs. RAG",
    ACL 2025.

Key formulas:
    BC(q, d) = ppl(q | d) / ppl(q)                     — Eq. (1)
    Edge(q, d) = (ppl(q) - ppl(q | d)) / ppl(q)        — Eq. (2)
    CS(G) = -Σ (h_i / 2m) · log2(h_i / 2m)             — Eq. (3)

BC measures boundary separation between chunks. A BC approaching 1.0 means
chunks are independent (good boundaries). Typical range: 0.77–0.87.

CS measures structural entropy of the semantic association graph. Lower CS
means chunks are more independent (better chunking). Typical ranges:
    CS_c (cross-chunk): 1.9–2.4
    CS_i (intra-document): 1.1–1.9

The module is model-agnostic: perplexity computation is provided via a
callable ``ppl_fn(text, context=None) -> float``.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

class PplFn(Protocol):
    """Perplexity function signature.

    Args:
        text: The text whose perplexity to compute.
        context: Optional conditioning context. When None, compute
            unconditional ppl(text). When provided, compute ppl(text | context).

    Returns:
        Perplexity as a positive float.
    """

    def __call__(self, text: str, context: str | None = None) -> float: ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BCScore:
    """Boundary Clarity result for a pair of chunks.

    Attributes:
        bc: BC(q, d) = ppl(q | d) / ppl(q). Higher = better boundary.
        edge: Edge weight = 1 - BC. Range [0, 1]. High = strong correlation.
        ppl_q: Unconditional perplexity of the query chunk.
        ppl_q_given_d: Contrastive perplexity of q given context d.
    """
    bc: float
    edge: float
    ppl_q: float
    ppl_q_given_d: float


@dataclass
class CSScore:
    """Chunk Stickiness result for a set of chunks.

    Attributes:
        cs: Structural entropy of the semantic association graph.
            Lower = better (chunks are more independent).
        n_nodes: Number of nodes (chunks) in the graph.
        n_edges: Number of edges above the threshold.
        threshold_k: Edge weight threshold K used for graph construction.
        skip_delta: Sequential skip constraint δ.
    """
    cs: float
    n_nodes: int
    n_edges: int
    threshold_k: float
    skip_delta: int


@dataclass
class ChunkingMetrics:
    """Combined BC and CS evaluation result.

    Attributes:
        bc_mean: Mean BC across all evaluated chunk pairs.
        bc_scores: Per-pair BC scores (optional, can be large).
        cs: CS score for the chunk set.
    """
    bc_mean: float
    bc_scores: list[BCScore] = field(default_factory=list)
    cs: CSScore | None = None


# ---------------------------------------------------------------------------
# BC computation
# ---------------------------------------------------------------------------

def calculate_bc(
    ppl_q: float,
    ppl_q_given_d: float,
) -> BCScore:
    """Compute Boundary Clarity from pre-computed perplexity values.

    BC(q, d) = ppl(q | d) / ppl(q)                 — Eq. (1)
    Edge(q, d) = (ppl(q) - ppl(q | d)) / ppl(q)    — Eq. (2)

    Args:
        ppl_q: Unconditional perplexity ppl(q).
        ppl_q_given_d: Contrastive perplexity ppl(q | d).

    Returns:
        BCScore with bc, edge, and the raw perplexity values.

    Raises:
        ValueError: If perplexity values are not positive.
    """
    if ppl_q <= 0 or ppl_q_given_d <= 0:
        raise ValueError(
            f"Perplexity values must be positive, got ppl_q={ppl_q}, "
            f"ppl_q_given_d={ppl_q_given_d}"
        )

    bc = ppl_q_given_d / ppl_q
    edge = (ppl_q - ppl_q_given_d) / ppl_q  # = 1 - bc

    return BCScore(bc=bc, edge=edge, ppl_q=ppl_q, ppl_q_given_d=ppl_q_given_d)


def calculate_bc_from_fn(
    query_chunk: str,
    context_chunk: str,
    ppl_fn: PplFn,
) -> BCScore:
    """Compute BC by calling the perplexity function.

    Args:
        query_chunk: Text q whose boundary clarity to measure.
        context_chunk: Text d used as conditioning context.
        ppl_fn: Model-agnostic perplexity function.

    Returns:
        BCScore for the (q, d) pair.
    """
    ppl_q = ppl_fn(query_chunk)
    ppl_q_given_d = ppl_fn(query_chunk, context=context_chunk)
    return calculate_bc(ppl_q, ppl_q_given_d)


# ---------------------------------------------------------------------------
# CS computation
# ---------------------------------------------------------------------------

def _structural_entropy(degrees: list[int], total_edges: int) -> float:
    """Compute structural entropy (1D) of a graph.

    CS(G) = -Σ (h_i / 2m) · log2(h_i / 2m)    — Eq. (3)

    where h_i = degree of node i, m = number of edges.

    Args:
        degrees: Degree of each node.
        total_edges: Total number of edges m in the graph.

    Returns:
        Structural entropy. 0.0 if graph has no edges.
    """
    if total_edges == 0:
        return 0.0

    two_m = 2 * total_edges
    entropy = 0.0
    for h_i in degrees:
        if h_i > 0:
            p = h_i / two_m
            entropy -= p * math.log2(p)

    return entropy


def calculate_cs(
    edge_weights: list[list[float]],
    threshold_k: float = 0.1,
    skip_delta: int = 1,
) -> CSScore:
    """Compute Chunk Stickiness from a pre-computed edge weight matrix.

    Constructs a semantic association graph:
    - Nodes = chunks
    - Edge (i, j) exists iff edge_weights[i][j] > K and |j - i| > δ

    The sequential constraint (|j - i| > δ) skips adjacent chunks since
    adjacency alone does not imply semantic dependence.

    Args:
        edge_weights: N×N matrix of edge weights (from Eq. 2).
            edge_weights[i][j] = Edge(d_i, d_j).
        threshold_k: Minimum edge weight to include in graph.
        skip_delta: Minimum index distance between nodes for an edge.

    Returns:
        CSScore with structural entropy and graph statistics.
    """
    n = len(edge_weights)
    if n == 0:
        return CSScore(cs=0.0, n_nodes=0, n_edges=0,
                        threshold_k=threshold_k, skip_delta=skip_delta)

    degrees = [0] * n
    total_edges = 0

    for i in range(n):
        for j in range(i + 1, n):
            if abs(j - i) <= skip_delta:
                continue
            # Use max of both directions for undirected graph
            w = max(edge_weights[i][j], edge_weights[j][i])
            if w > threshold_k:
                degrees[i] += 1
                degrees[j] += 1
                total_edges += 1

    cs = _structural_entropy(degrees, total_edges)

    return CSScore(
        cs=cs,
        n_nodes=n,
        n_edges=total_edges,
        threshold_k=threshold_k,
        skip_delta=skip_delta,
    )


def calculate_cs_from_fn(
    chunks: list[str],
    ppl_fn: PplFn,
    threshold_k: float = 0.1,
    skip_delta: int = 1,
) -> tuple[CSScore, list[list[float]]]:
    """Compute CS by calling the perplexity function for all chunk pairs.

    Args:
        chunks: List of chunk texts.
        ppl_fn: Model-agnostic perplexity function.
        threshold_k: Edge weight threshold K.
        skip_delta: Sequential skip constraint δ.

    Returns:
        Tuple of (CSScore, edge_weight_matrix).
    """
    n = len(chunks)

    # Pre-compute unconditional perplexities
    ppl_solo: list[float] = [ppl_fn(c) for c in chunks]

    # Compute edge weights for all pairs with |j-i| > delta
    edge_weights = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or abs(j - i) <= skip_delta:
                continue
            ppl_cond = ppl_fn(chunks[i], context=chunks[j])
            edge_weights[i][j] = (ppl_solo[i] - ppl_cond) / ppl_solo[i]

    cs = calculate_cs(edge_weights, threshold_k=threshold_k, skip_delta=skip_delta)
    return cs, edge_weights


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def evaluate_chunking(
    chunks: list[str],
    ppl_fn: PplFn,
    threshold_k: float = 0.1,
    skip_delta: int = 1,
    bc_pairs: list[tuple[int, int]] | None = None,
    include_all_bc: bool = False,
) -> ChunkingMetrics:
    """Evaluate chunking quality using both BC and CS metrics.

    Args:
        chunks: List of chunk texts to evaluate.
        ppl_fn: Model-agnostic perplexity function.
        threshold_k: Edge weight threshold K for CS graph construction.
        skip_delta: Sequential skip δ — minimum index gap for CS edges.
        bc_pairs: Specific (i, j) pairs to compute BC for. If None, uses
            adjacent pairs (i, i+1) for BC measurement.
        include_all_bc: If True, store all individual BCScores in the result.

    Returns:
        ChunkingMetrics with mean BC and CS score.
    """
    if len(chunks) < 2:
        logger.warning("Need at least 2 chunks for meaningful metrics.")
        return ChunkingMetrics(bc_mean=1.0)

    # --- BC ---
    if bc_pairs is None:
        # Default: adjacent pairs
        bc_pairs = [(i, i + 1) for i in range(len(chunks) - 1)]

    bc_scores: list[BCScore] = []
    for i, j in bc_pairs:
        score = calculate_bc_from_fn(chunks[i], chunks[j], ppl_fn)
        bc_scores.append(score)

    bc_mean = sum(s.bc for s in bc_scores) / len(bc_scores) if bc_scores else 1.0

    # --- CS ---
    cs, _ = calculate_cs_from_fn(
        chunks, ppl_fn, threshold_k=threshold_k, skip_delta=skip_delta,
    )

    return ChunkingMetrics(
        bc_mean=bc_mean,
        bc_scores=bc_scores if include_all_bc else [],
        cs=cs,
    )


# ---------------------------------------------------------------------------
# vLLM perplexity client
# ---------------------------------------------------------------------------

def create_ppl_client(
    base_url: str = "http://localhost:8000/v1",
    model: str = "default",
    api_key: str = "EMPTY",
    max_tokens: int = 1,
    timeout: float = 60.0,
) -> PplFn:
    """Create a perplexity function using vLLM's OpenAI-compatible API.

    Uses the completions endpoint with ``echo=True`` and ``logprobs`` to
    compute perplexity from log-probabilities of each token.

    Perplexity is computed as:
        ppl = exp(-1/N · Σ log P(t_i | t_<i))

    where the sum is over the tokens of ``text`` (not the context prefix).

    Args:
        base_url: vLLM OpenAI-compatible API base URL.
        model: Model name/ID for the API.
        api_key: API key (vLLM typically uses "EMPTY").
        max_tokens: Max tokens to generate (1 since we only need logprobs).
        timeout: HTTP request timeout in seconds.

    Returns:
        A callable matching the PplFn protocol.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise ImportError(
            "openai package required for create_ppl_client. "
            "Install with: pip install openai"
        ) from e

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def ppl_fn(text: str, context: str | None = None) -> float:
        """Compute perplexity of text, optionally conditioned on context.

        Uses the completions API with echo=True to get logprobs for each
        token in the prompt. When context is provided, only the logprobs
        of the text tokens (after the context prefix) are used.
        """
        if not text.strip():
            raise ValueError("Cannot compute perplexity of empty text.")

        if context is not None:
            prompt = context + "\n\n" + text
        else:
            prompt = text

        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            echo=True,
            logprobs=1,
            temperature=0.0,
        )

        choice = response.choices[0]
        tokens = choice.logprobs.tokens
        token_logprobs = choice.logprobs.token_logprobs

        if context is not None:
            # Find where context ends and text begins.
            # Tokenize context alone to determine the split point.
            ctx_response = client.completions.create(
                model=model,
                prompt=context,
                max_tokens=max_tokens,
                echo=True,
                logprobs=1,
                temperature=0.0,
            )
            n_ctx_tokens = len(ctx_response.choices[0].logprobs.tokens)
            # Skip context tokens; the separator "\n\n" may add 1-2 tokens,
            # but the key insight is: we only want logprobs for text tokens.
            # Use n_ctx_tokens as the split point (conservative: includes
            # separator tokens in context, so text logprobs are clean).
            start_idx = n_ctx_tokens
        else:
            # First token has no logprob (it's the BOS/first token).
            start_idx = 1

        # Collect logprobs for the text portion
        text_logprobs = [
            lp for lp in token_logprobs[start_idx:]
            if lp is not None
        ]

        if not text_logprobs:
            logger.warning(
                "No logprobs found for text tokens. Returning high perplexity."
            )
            return 1e6

        avg_neg_logprob = -sum(text_logprobs) / len(text_logprobs)
        perplexity = math.exp(avg_neg_logprob)

        return perplexity

    return ppl_fn


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

@dataclass
class SensitivityResult:
    """Result of a single (K, δ) configuration in sensitivity analysis.

    Attributes:
        threshold_k: Edge weight threshold K.
        skip_delta: Sequential skip constraint δ.
        cs: Structural entropy value.
        n_edges: Number of edges in the constructed graph.
        graph_density: Fraction of possible edges that exist.
        bc_mean: Mean BC (same across K/δ since BC is pairwise).
    """
    threshold_k: float
    skip_delta: int
    cs: float
    n_edges: int
    graph_density: float
    bc_mean: float


def sensitivity_analysis(
    edge_weights: list[list[float]],
    bc_mean: float,
    k_values: list[float] | None = None,
    delta_values: list[int] | None = None,
) -> list[SensitivityResult]:
    """Run CS sensitivity analysis over a grid of (K, δ) values.

    BC is independent of K and δ (it's computed per adjacent pair),
    so bc_mean is repeated for reference. CS changes with each (K, δ).

    This function reuses a pre-computed edge_weight matrix, so it
    does NOT require additional LLM calls. Call calculate_cs_from_fn()
    first to get the matrix, then pass it here.

    Args:
        edge_weights: N×N edge weight matrix from calculate_cs_from_fn().
        bc_mean: Pre-computed mean BC value.
        k_values: Threshold K values to sweep. Default [0.05, 0.1, 0.15, 0.2].
        delta_values: Skip δ values to sweep. Default [1, 2, 3].

    Returns:
        List of SensitivityResult, one per (K, δ) combination.
    """
    if k_values is None:
        k_values = [0.05, 0.1, 0.15, 0.2]
    if delta_values is None:
        delta_values = [1, 2, 3]

    n = len(edge_weights)
    # Max possible edges (undirected, excluding self-loops)
    max_possible = n * (n - 1) // 2

    results: list[SensitivityResult] = []
    for k in k_values:
        for delta in delta_values:
            cs_score = calculate_cs(edge_weights, threshold_k=k, skip_delta=delta)
            density = cs_score.n_edges / max_possible if max_possible > 0 else 0.0
            results.append(SensitivityResult(
                threshold_k=k,
                skip_delta=delta,
                cs=cs_score.cs,
                n_edges=cs_score.n_edges,
                graph_density=density,
                bc_mean=bc_mean,
            ))

    return results


def format_sensitivity_table(
    results: list[SensitivityResult],
    model_name: str = "",
) -> str:
    """Format sensitivity analysis results as a markdown table.

    Produces a table suitable for paper appendix:

        | K    | δ | CS    | Edges | Density | BC_mean |
        |------|---|-------|-------|---------|---------|
        | 0.05 | 1 | 2.341 | 45    | 0.312   | 0.823   |

    Args:
        results: Output from sensitivity_analysis().
        model_name: Optional model name for the table header.

    Returns:
        Markdown-formatted table string.
    """
    header = f"### Sensitivity Analysis{f' — {model_name}' if model_name else ''}\n\n"
    header += "| K | δ | CS | Edges | Density | BC_mean |\n"
    header += "|------|---|-------|-------|---------|--------|\n"

    rows = []
    for r in results:
        rows.append(
            f"| {r.threshold_k:.2f} | {r.skip_delta} "
            f"| {r.cs:.3f} | {r.n_edges:>5} "
            f"| {r.graph_density:.3f} | {r.bc_mean:.3f} |"
        )

    return header + "\n".join(rows)


def compare_models_sensitivity(
    model_results: dict[str, list[SensitivityResult]],
) -> str:
    """Compare sensitivity results across multiple models.

    Produces a table showing CS delta between models at each (K, δ):

        | K    | δ | Model_A CS | Model_B CS | Δ(B-A) |
        |------|---|-----------|-----------|--------|

    Args:
        model_results: Dict mapping model_name → sensitivity results.
            Must have the same (K, δ) grid for all models.

    Returns:
        Markdown-formatted comparison table string.
    """
    model_names = list(model_results.keys())
    if len(model_names) < 2:
        return format_sensitivity_table(
            list(model_results.values())[0], model_names[0]
        )

    # Index by (K, δ) for each model
    indexed: dict[str, dict[tuple[float, int], SensitivityResult]] = {}
    for name, results in model_results.items():
        indexed[name] = {(r.threshold_k, r.skip_delta): r for r in results}

    # Get all (K, δ) keys from first model
    keys = sorted(indexed[model_names[0]].keys())

    # Build header
    cs_cols = " | ".join(f"{name} CS" for name in model_names)
    header = f"### Model Comparison — CS Sensitivity\n\n"
    header += f"| K | δ | {cs_cols} | Δ (improvement) |\n"
    sep = "|------|---|" + "|".join(["----------"] * len(model_names)) + "|----------------|\n"
    header += sep

    rows = []
    for k, delta in keys:
        cs_values = []
        for name in model_names:
            r = indexed[name].get((k, delta))
            cs_values.append(r.cs if r else float("nan"))

        cs_strs = " | ".join(f"{v:>9.3f}" for v in cs_values)
        # Delta: last model minus first model (negative = improvement)
        delta_val = cs_values[-1] - cs_values[0]
        sign = "" if delta_val >= 0 else ""
        rows.append(f"| {k:.2f} | {delta} | {cs_strs} | {sign}{delta_val:+.3f} |")

    return header + "\n".join(rows)


__all__ = [
    "BCScore",
    "CSScore",
    "ChunkingMetrics",
    "SensitivityResult",
    "PplFn",
    "calculate_bc",
    "calculate_bc_from_fn",
    "calculate_cs",
    "calculate_cs_from_fn",
    "evaluate_chunking",
    "create_ppl_client",
    "sensitivity_analysis",
    "format_sensitivity_table",
    "compare_models_sensitivity",
]
