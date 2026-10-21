#!/usr/bin/env python3
"""Batch GT validation using a judge LLM model (text-based).

Usage:
    # Validate all Korean documents (10% sampling)
    python scripts/validate_gt.py --dataset documents

    # Validate a specific document with higher sampling
    python scripts/validate_gt.py --dataset documents --doc-id kogov_001 --sample-ratio 0.3

    # Validate English papers
    python scripts/validate_gt.py --dataset papers

    # Use a specific model
    python scripts/validate_gt.py --dataset documents --model qwen3.5-122b
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.gt_validator import validate_document, result_to_dict


DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3.5-122b"


def main():
    parser = argparse.ArgumentParser(description="Validate pseudo GT quality with judge LLM")
    parser.add_argument("--dataset", choices=["documents", "papers"], default="documents")
    parser.add_argument("--doc-id", type=str, help="Specific document ID (e.g. kogov_001)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Judge LLM API endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Judge LLM model name")
    parser.add_argument("--sample-ratio", type=float, default=0.1,
                        help="Fraction of pages to sample (default: 0.1 = 10%%)")
    parser.add_argument("--timeout", type=float, default=180.0, help="API timeout (seconds)")
    parser.add_argument("--batch-size", type=int, default=2, help="Concurrent validation requests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    dataset_dir = Path("datasets") / args.dataset

    if not dataset_dir.exists():
        print(f"Dataset not found: {dataset_dir}")
        return

    # Collect target documents
    if args.doc_id:
        doc_dirs = [dataset_dir / args.doc_id]
        if not doc_dirs[0].exists():
            print(f"Document not found: {doc_dirs[0]}")
            return
    else:
        doc_dirs = sorted(d for d in dataset_dir.iterdir() if d.is_dir())

    # Filter to those with gt_pages/
    targets = [d for d in doc_dirs if (d / "gt_pages").exists()]

    if not targets:
        print("No documents with GT to validate.")
        return

    print(f"\n{'=' * 60}")
    print(f"GT Quality Validation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Documents: {len(targets)}")
    print(f"  Judge model: {args.model}")
    print(f"  API: {args.api_url}")
    print(f"  Sample ratio: {args.sample_ratio:.0%}")
    print(f"  Batch size: {args.batch_size}")
    print(f"{'=' * 60}\n")

    total_start = time.time()
    all_results = []

    for doc_dir in targets:
        print(f"\n--- {doc_dir.name} ---")
        result = validate_document(
            doc_dir=doc_dir,
            api_url=args.api_url,
            model=args.model,
            timeout=args.timeout,
            sample_ratio=args.sample_ratio,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_results.append(result)

    total_time = time.time() - total_start

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Document':<15} {'Sampled':>8} {'Avg':>6} {'Min':>5} {'Max':>5} {'Pass%':>7} {'Failed':>8}")
    print(f"{'-' * 15} {'-' * 8} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 7} {'-' * 8}")

    total_sampled = 0
    total_failed = 0
    all_scores = []

    for r in all_results:
        total_sampled += r.sampled_pages
        total_failed += len(r.failed_pages)
        valid = [pr for pr in r.page_results if pr.error is None]
        all_scores.extend(pr.score for pr in valid)

        failed_str = str(r.failed_pages) if r.failed_pages else "none"
        print(f"{r.doc_id:<15} {r.sampled_pages:>8} {r.avg_score:>6.2f} "
              f"{r.min_score:>5} {r.max_score:>5} {r.acceptable_ratio:>6.0%} "
              f"{failed_str:>8}")

    print(f"{'-' * 15} {'-' * 8} {'-' * 6} {'-' * 5} {'-' * 5} {'-' * 7} {'-' * 8}")
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    overall_pass = sum(1 for s in all_scores if s >= 3) / len(all_scores) if all_scores else 0
    print(f"{'TOTAL':<15} {total_sampled:>8} {overall_avg:>6.2f} "
          f"{'':>5} {'':>5} {overall_pass:>6.0%} {total_failed:>8}")
    print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f}m)")

    # Save report
    report = {
        "dataset": args.dataset,
        "judge_model": args.model,
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "total_documents": len(all_results),
        "total_sampled_pages": total_sampled,
        "overall_avg_score": round(overall_avg, 2),
        "overall_pass_rate": round(overall_pass, 3),
        "total_failed_pages": total_failed,
        "total_time_seconds": round(total_time, 1),
        "documents": [result_to_dict(r) for r in all_results],
    }

    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport saved: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
