"""Evaluation metrics for document parsing quality."""

from evaluation.metrics.cer import calculate_cer, calculate_wer, calculate_edit_distance
from evaluation.metrics.structure import (
    calculate_structure_eval,
    calculate_structure_f1,  # legacy wrapper
    StructureEvalResult,
    StructureF1Result,  # legacy alias
    extract_elements,
)
from evaluation.metrics.chunking import (
    calculate_bc,
    calculate_cs,
    evaluate_chunking,
    create_ppl_client,
    BCScore,
    CSScore,
    ChunkingMetrics,
)
from evaluation.metrics.teds import calculate_teds
from evaluation.metrics.retrieval import (
    ndcg_at_k,
    hit_at_k,
    mrr_at_k,
    recall_at_k,
    evaluate_retrieval,
    RetrievalMetrics,
)

__all__ = [
    "calculate_cer", "calculate_wer", "calculate_edit_distance",
    "calculate_bc", "calculate_cs", "evaluate_chunking",
    "create_ppl_client", "BCScore", "CSScore", "ChunkingMetrics",
    "calculate_structure_eval", "calculate_structure_f1",
    "StructureEvalResult", "StructureF1Result",
    "extract_elements",
    "calculate_teds",
    "ndcg_at_k", "hit_at_k", "mrr_at_k", "recall_at_k",
    "evaluate_retrieval", "RetrievalMetrics",
]
