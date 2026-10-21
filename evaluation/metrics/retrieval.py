"""Retrieval evaluation metrics following the BEIR benchmark.

Implements nDCG@k, Hit@K, MRR@K, and Recall@K as defined in:

    Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021).
    BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information
    Retrieval Models. In *Proceedings of NeurIPS 2021 Datasets and Benchmarks
    Track*. https://arxiv.org/abs/2104.08663

Input format follows the standard BEIR / pytrec_eval convention:
    qrels:   dict[str, dict[str, int]]   — query_id -> {doc_id: relevance}
    results: dict[str, dict[str, float]] — query_id -> {doc_id: score}
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, Dict[str, float]]


# ---------------------------------------------------------------------------
# Dataclass for bundled results
# ---------------------------------------------------------------------------
@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation results across multiple k values.

    Attributes:
        ndcg: nDCG@k values keyed by k (e.g. ``{10: 0.45}``).
        hit: Hit@K values keyed by k.
        mrr: MRR@K values keyed by k.
        recall: Recall@K values keyed by k.
        k_values: The k values that were evaluated.
    """

    ndcg: Dict[int, float] = field(default_factory=dict)
    hit: Dict[int, float] = field(default_factory=dict)
    mrr: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    k_values: List[int] = field(default_factory=list)

    # Convenience ---------------------------------------------------------

    def flat_dict(self) -> Dict[str, float]:
        """Return a flat dictionary suitable for logging / JSON export.

        Keys follow the pattern ``metric@k``, e.g. ``"nDCG@10"``.
        """
        out: Dict[str, float] = {}
        for k in self.k_values:
            out[f"nDCG@{k}"] = self.ndcg.get(k, 0.0)
            out[f"Hit@{k}"] = self.hit.get(k, 0.0)
            out[f"MRR@{k}"] = self.mrr.get(k, 0.0)
            out[f"Recall@{k}"] = self.recall.get(k, 0.0)
        return out


# ---------------------------------------------------------------------------
# Helper: rank a single query's results
# ---------------------------------------------------------------------------
def _ranked_doc_ids(scores: Dict[str, float], k: int) -> List[str]:
    """Return the top-k doc_ids sorted by descending score."""
    return [
        doc_id
        for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    ]


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(
    qrels: Qrels,
    results: Results,
    k: int = 10,
) -> float:
    """Compute mean normalised Discounted Cumulative Gain at rank *k*.

    DCG@k  = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k

    where IDCG@k is the DCG of the ideal (perfect) ranking.  Queries that
    have no relevant documents in *qrels* are excluded from the average,
    matching the convention used by BEIR.

    Reference:
        Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot
        Evaluation of Information Retrieval Models", NeurIPS 2021.
    """
    scores: List[float] = []

    for qid, doc_rels in qrels.items():
        # Skip queries with no positive relevance judgements.
        if not any(r > 0 for r in doc_rels.values()):
            continue

        query_results = results.get(qid, {})
        ranked = _ranked_doc_ids(query_results, k)

        # DCG of the retrieved ranking.
        dcg = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            rel = doc_rels.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 1)

        # IDCG: sort all true relevances descending, take top-k.
        ideal_rels = sorted(doc_rels.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 1)
            for i, rel in enumerate(ideal_rels, start=1)
        )

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def hit_at_k(
    qrels: Qrels,
    results: Results,
    k: int = 10,
) -> float:
    """Compute Hit@K averaged across queries.

    Hit@K for a single query is 1 if *any* document with relevance > 0
    appears in the top-K retrieved results, and 0 otherwise.
    """
    hits: List[float] = []

    for qid, doc_rels in qrels.items():
        if not any(r > 0 for r in doc_rels.values()):
            continue

        query_results = results.get(qid, {})
        ranked = _ranked_doc_ids(query_results, k)
        hit = 1.0 if any(doc_rels.get(doc_id, 0) > 0 for doc_id in ranked) else 0.0
        hits.append(hit)

    return sum(hits) / len(hits) if hits else 0.0


def mrr_at_k(
    qrels: Qrels,
    results: Results,
    k: int = 10,
) -> float:
    """Compute Mean Reciprocal Rank at rank *k*.

    MRR@K = (1 / |Q|) * sum_{q in Q} 1 / rank_q

    where rank_q is the position of the first relevant document (relevance > 0)
    within the top-K results.  If no relevant document appears in the top-K,
    the reciprocal rank for that query is 0.
    """
    rr_values: List[float] = []

    for qid, doc_rels in qrels.items():
        if not any(r > 0 for r in doc_rels.values()):
            continue

        query_results = results.get(qid, {})
        ranked = _ranked_doc_ids(query_results, k)

        rr = 0.0
        for i, doc_id in enumerate(ranked, start=1):
            if doc_rels.get(doc_id, 0) > 0:
                rr = 1.0 / i
                break
        rr_values.append(rr)

    return sum(rr_values) / len(rr_values) if rr_values else 0.0


def recall_at_k(
    qrels: Qrels,
    results: Results,
    k: int = 10,
) -> float:
    """Compute mean Recall at rank *k*.

    R@K = |{relevant docs in top-K}| / |{all relevant docs}|

    Averaged across all queries that have at least one relevant document.
    """
    recall_values: List[float] = []

    for qid, doc_rels in qrels.items():
        relevant_docs = {did for did, rel in doc_rels.items() if rel > 0}
        if not relevant_docs:
            continue

        query_results = results.get(qid, {})
        ranked = set(_ranked_doc_ids(query_results, k))

        recall_values.append(len(ranked & relevant_docs) / len(relevant_docs))

    return sum(recall_values) / len(recall_values) if recall_values else 0.0


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_retrieval(
    qrels: Qrels,
    results: Results,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> RetrievalMetrics:
    """Evaluate retrieval quality using all metrics at the given k values.

    This mirrors the evaluation protocol of the BEIR benchmark (Thakur et al.,
    NeurIPS 2021), computing nDCG@k, Hit@K, MRR@K, and Recall@K for each
    requested cutoff.

    Args:
        qrels: Ground-truth relevance judgements.
            ``{query_id: {doc_id: int_relevance, ...}, ...}``
        results: Retrieval scores (higher = more relevant).
            ``{query_id: {doc_id: float_score, ...}, ...}``
        k_values: Rank cutoffs to evaluate at.

    Returns:
        A :class:`RetrievalMetrics` dataclass with per-k results.
    """
    k_list = sorted(k_values)
    metrics = RetrievalMetrics(k_values=k_list)

    for k in k_list:
        metrics.ndcg[k] = ndcg_at_k(qrels, results, k)
        metrics.hit[k] = hit_at_k(qrels, results, k)
        metrics.mrr[k] = mrr_at_k(qrels, results, k)
        metrics.recall[k] = recall_at_k(qrels, results, k)

    return metrics
