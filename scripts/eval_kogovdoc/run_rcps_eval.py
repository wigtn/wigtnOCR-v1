#!/usr/bin/env python3
"""KoGovDoc retrieval evaluation (Hit@K, MRR@K, nDCG@K, RCPS) for one parser output dir.

This is the protocol of the RCPS paper (EMNLP 2026 Industry Track), run through its
reference implementation in wigtn/WigtnOCR-RADP:

    - 663 Q–A pairs (data/KoGovDoc-RAG/qa_pairs_v1.jsonl in WigtnOCR-RADP)
    - 294 pages = the Wigtn/KoGovDoc-Bench validation split (val.jsonl)
    - a chunk is relevant only if it comes from the reference page and contains
      the normalised reference answer
    - retrievers BGE-M3, multilingual-e5-large, Qwen3-Embedding-8B; k in {1, 5, 10}
    - RCPS = mean MRR@k over every (retriever, k) pair

It does not reproduce the v1 KoGovDoc table in the README (Hit@1 0.739); that table comes from
run_retrieval_eval.py, which uses a different query set and relevance rule.

Usage:
    pip install -r requirements-eval.txt
    python scripts/eval_kogovdoc/run_rcps_eval.py \
        --pred-dir results/kogovdoc/wigtnocr-2b/predictions \
        --val-jsonl datasets/kogovdoc-bench/val.jsonl \
        --qa third_party/WigtnOCR-RADP/data/KoGovDoc-RAG/qa_pairs_v1.jsonl \
        --device cuda --out results/kogovdoc/wigtnocr-2b_rcps.json
"""

import argparse
import json
import logging
from pathlib import Path

from wigtnocr_radp.evaluation import (
    BgeM3Retriever,
    FixedSizeChunker,
    MarkdownHeaderChunker,
    ParserNativeChunker,
    compute_rcps,
)
from wigtnocr_radp.evaluation.parser_outputs import load_parser_outputs
from wigtnocr_radp.evaluation.rcps import load_qa_pairs
from wigtnocr_radp.evaluation.retrievers import (
    MultilingualE5LargeRetriever,
    Qwen3EmbeddingRetriever,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Same chunker settings as WigtnOCR-RADP scripts/evaluation/baseline_grid.py
CHUNKERS = {
    "parser_native": lambda: ParserNativeChunker(min_chars=30),
    "md_h3": lambda: MarkdownHeaderChunker(max_level=3),
    "fixed500": lambda: FixedSizeChunker(size=500),
}

RETRIEVERS = {
    "bge-m3": lambda device: BgeM3Retriever(device=device, batch_size=32),
    "ml-e5-large": lambda device: MultilingualE5LargeRetriever(device=device, batch_size=32),
    "qwen3-emb-8b": lambda device: Qwen3EmbeddingRetriever(device=device, batch_size=8),
}


def main():
    parser = argparse.ArgumentParser(description="KoGovDoc retrieval evaluation (RCPS protocol)")
    parser.add_argument("--pred-dir", type=Path, required=True,
                        help="Directory of <doc_id>_<page_stem>.md parser outputs")
    parser.add_argument("--val-jsonl", type=Path, required=True,
                        help="Wigtn/KoGovDoc-Bench val.jsonl (defines the 294 page IDs)")
    parser.add_argument("--qa", type=Path, required=True,
                        help="WigtnOCR-RADP data/KoGovDoc-RAG/qa_pairs_v1.jsonl")
    parser.add_argument("--retrievers", default="bge-m3,ml-e5-large,qwen3-emb-8b",
                        help="Comma-separated subset of: " + ", ".join(RETRIEVERS))
    parser.add_argument("--chunker", choices=tuple(CHUNKERS), default="parser_native")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    qa_pairs = load_qa_pairs(args.qa)
    pages = load_parser_outputs(args.pred_dir, args.val_jsonl)
    retrievers = [RETRIEVERS[name](args.device) for name in args.retrievers.split(",")]

    res = compute_rcps(
        qa_pairs=qa_pairs,
        parser_pages=pages,
        retrievers=retrievers,
        chunker=CHUNKERS[args.chunker](),
        k_values=(1, 5, 10),
    )

    names = [r.name for r in retrievers]
    by_retriever = res["by_retriever"]

    def mean_over_retrievers(metric: str, k: int) -> float:
        return sum(by_retriever[n][metric][k] for n in names) / len(names)

    summary = {
        "pred_dir": str(args.pred_dir),
        "num_qa": len(qa_pairs),
        "num_pages": len(pages),
        "num_chunks": res["meta"]["num_chunks"],
        "chunker": res["meta"]["chunker"],
        "retrievers": names,
        "rcps": res["rcps"],
        "hit@1": mean_over_retrievers("hit", 1),
        "hit@5": mean_over_retrievers("hit", 5),
        "hit@10": mean_over_retrievers("hit", 10),
        "mrr@10": mean_over_retrievers("mrr", 10),
        "ndcg@10": mean_over_retrievers("ndcg", 10),
        "by_retriever": by_retriever,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\npages={summary['num_pages']} chunks={summary['num_chunks']} "
          f"retrievers={'+'.join(names)} chunker={summary['chunker']}")
    print(f"RCPS={summary['rcps']:.4f}  Hit@1={summary['hit@1']:.4f}  Hit@5={summary['hit@5']:.4f}  "
          f"Hit@10={summary['hit@10']:.4f}  MRR@10={summary['mrr@10']:.4f}  "
          f"nDCG@10={summary['ndcg@10']:.4f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
