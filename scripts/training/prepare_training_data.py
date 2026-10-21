#!/usr/bin/env python3
"""Prepare training data for ms-swift LoRA fine-tuning.

Converts validated GT pages into ms-swift JSONL format with rendered page images.

Usage:
    # Prepare data from both datasets
    python scripts/prepare_training_data.py --dataset all

    # Prepare only documents
    python scripts/prepare_training_data.py --dataset documents

    # Custom min score and sampling
    python scripts/prepare_training_data.py --dataset all --min-score 4 --max-doc-ratio 0.2

    # Dry run (show stats without rendering images)
    python scripts/prepare_training_data.py --dataset all --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.data_prep import prepare_training_data, _load_valid_pages, _downsample_documents


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for ms-swift")
    parser.add_argument("--dataset", choices=["documents", "papers", "all"],
                        default="all", help="Which datasets to include")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/training"),
                        help="Output directory for JSONL and images")
    parser.add_argument("--dpi", type=int, default=200, help="Image rendering DPI")
    parser.add_argument("--min-score", type=int, default=3,
                        help="Minimum validation score to include (1-5)")
    parser.add_argument("--max-doc-ratio", type=float, default=0.25,
                        help="Max fraction any single document can represent")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show stats without generating data")
    args = parser.parse_args()

    datasets = ["documents", "papers"] if args.dataset == "all" else [args.dataset]
    dataset_dirs = [Path("datasets") / ds for ds in datasets]

    for d in dataset_dirs:
        if not d.exists():
            print(f"Dataset not found: {d}")
            return

    if args.dry_run:
        print(f"\n{'=' * 50}")
        print("DRY RUN — Data Statistics")
        print(f"{'=' * 50}")
        for dataset_dir in dataset_dirs:
            pages_map = _load_valid_pages(dataset_dir, min_score=args.min_score)
            total = sum(len(p) for p in pages_map.values())
            print(f"\n--- {dataset_dir.name} ---")
            print(f"  Documents: {len(pages_map)}")
            print(f"  Valid pages (score>={args.min_score}): {total}")
            for doc_id, pages in sorted(pages_map.items()):
                print(f"    {doc_id}: {len(pages)}p")

            pages_map = _downsample_documents(pages_map, max_ratio=args.max_doc_ratio, seed=args.seed)
            total_after = sum(len(p) for p in pages_map.values())
            print(f"  After downsampling (max_ratio={args.max_doc_ratio:.0%}): {total_after}")
        return

    stats = prepare_training_data(
        dataset_dirs=dataset_dirs,
        output_dir=args.output_dir,
        dpi=args.dpi,
        min_score=args.min_score,
        max_doc_ratio=args.max_doc_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"\nStats saved: {args.output_dir / 'data_stats.json'}")


if __name__ == "__main__":
    main()
