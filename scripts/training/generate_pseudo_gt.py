#!/usr/bin/env python3
"""Batch pseudo GT generation for documents.

Usage:
    # Generate GT for all Korean documents (small ones first)
    python scripts/generate_pseudo_gt.py --dataset documents

    # Generate GT for a specific document
    python scripts/generate_pseudo_gt.py --dataset documents --doc-id kogov_001

    # Generate GT for English papers
    python scripts/generate_pseudo_gt.py --dataset papers

    # Limit pages per document (for testing)
    python scripts/generate_pseudo_gt.py --dataset documents --doc-id kogov_002 --max-pages 3

    # Force regeneration (ignore existing gt.md)
    python scripts/generate_pseudo_gt.py --dataset documents --force
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.gt_generator import generate_pseudo_gt


DEFAULT_API_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = "qwen3-vl-30b-thinking"


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo GT for documents")
    parser.add_argument("--dataset", choices=["documents", "papers"], default="documents")
    parser.add_argument("--doc-id", type=str, help="Specific document ID (e.g. kogov_001)")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="VLM API endpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="VLM model name")
    parser.add_argument("--dpi", type=int, default=200, help="Image rendering DPI")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per document")
    parser.add_argument("--timeout", type=float, default=180.0, help="API timeout (seconds)")
    parser.add_argument("--batch-size", type=int, default=4, help="Concurrent VLM requests")
    parser.add_argument("--force", action="store_true", help="Regenerate even if gt.md exists")
    parser.add_argument("--lang", choices=["ko", "en"], default=None,
                        help="Language (default: auto-detect from dataset)")
    args = parser.parse_args()

    # Auto-detect language from dataset
    if args.lang is None:
        args.lang = "en" if args.dataset == "papers" else "ko"

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

    # Filter to those with PDF (doc.pdf or paper.pdf)
    targets = []
    for d in doc_dirs:
        pdf = d / "doc.pdf"
        if not pdf.exists():
            pdf = d / "paper.pdf"
        if pdf.exists():
            targets.append((d, pdf))
        else:
            print(f"  [WARN] {d.name}: no PDF found, skipping")

    if not targets:
        print("No documents to process.")
        return

    print(f"\n{'=' * 60}")
    print(f"Pseudo GT Generation")
    print(f"  Dataset: {args.dataset}")
    print(f"  Documents: {len(targets)}")
    print(f"  Model: {args.model}")
    print(f"  API: {args.api_url}")
    print(f"  DPI: {args.dpi}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Language: {args.lang}")
    if args.max_pages:
        print(f"  Max pages: {args.max_pages}")
    print(f"{'=' * 60}\n")

    total_start = time.time()
    results = []

    for doc_dir, pdf_path in targets:
        print(f"\n--- {doc_dir.name} ---")
        result = generate_pseudo_gt(
            pdf_path=pdf_path,
            output_dir=doc_dir,
            api_url=args.api_url,
            model=args.model,
            dpi=args.dpi,
            max_tokens=8192,
            timeout=args.timeout,
            max_pages=args.max_pages,
            skip_existing=not args.force,
            batch_size=args.batch_size,
            lang=args.lang,
        )
        results.append(result)
        print(f"  -> {result.processed_pages}/{result.total_pages} pages, "
              f"{result.total_time:.1f}s, "
              f"failed: {result.failed_pages if result.failed_pages else 'none'}")

    total_time = time.time() - total_start
    total_pages = sum(r.processed_pages for r in results)
    total_failed = sum(len(r.failed_pages) for r in results)

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Total documents: {len(results)}")
    print(f"  Total pages processed: {total_pages}")
    print(f"  Total failures: {total_failed}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")
    if total_pages > 0:
        print(f"  Avg time per page: {total_time / total_pages:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
