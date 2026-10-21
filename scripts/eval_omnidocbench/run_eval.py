#!/usr/bin/env python3
"""Run OmniDocBench evaluation using internalized pipeline.

Evaluates VLM predictions against OmniDocBench GT using:
- NED (Normalized Edit Distance) for text — OmniDocBench (CVPR 2025)
- TEDS / TEDS-S for tables — Zhong et al. (2019)
- NED for formulas
- Reading Order NED

Usage:
    # Evaluate single parser
    python scripts/run_omnidocbench_eval.py \
        --pred-dir results/omnidocbench/vlm_2b \
        --name vlm_2b

    # Evaluate and compare multiple parsers
    python scripts/run_omnidocbench_eval.py \
        --pred-dir results/omnidocbench/vlm_2b \
                   results/omnidocbench/vlm_30b \
        --name vlm_2b vlm_30b \
        --compare
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.omnidocbench.evaluator import OmniDocBenchEvaluator


def main():
    parser = argparse.ArgumentParser(description="Run OmniDocBench evaluation")
    parser.add_argument(
        "--pred-dir",
        type=Path,
        nargs="+",
        required=True,
        help="Prediction directories (one per parser)",
    )
    parser.add_argument(
        "--name",
        nargs="+",
        required=True,
        help="Names for each parser",
    )
    parser.add_argument(
        "--gt-path",
        type=Path,
        default=Path("datasets/omnidocbench/OmniDocBench.json"),
        help="Path to OmniDocBench.json GT file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/omnidocbench"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print comparison table",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-page progress",
    )

    args = parser.parse_args()

    if len(args.pred_dir) != len(args.name):
        print("ERROR: --pred-dir and --name must have same count")
        sys.exit(1)

    all_results = {}

    for pred_dir, name in zip(args.pred_dir, args.name):
        print(f"\n{'='*60}")
        print(f"Evaluating: {name} ({pred_dir})")
        print(f"{'='*60}")

        md_count = len(list(pred_dir.glob("*.md")))
        if md_count == 0:
            print(f"WARNING: No .md files in {pred_dir}")
            continue
        print(f"Found {md_count} prediction files")

        # Run evaluation
        evaluator = OmniDocBenchEvaluator(
            gt_path=args.gt_path,
            pred_dir=pred_dir,
        )
        result = evaluator.evaluate(verbose=not args.quiet)
        evaluator.print_summary(result)

        # Save results
        result_dict = evaluator.to_dict(result)
        result_path = args.output_dir / f"{name}_results.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        print(f"Results saved: {result_path}")

        all_results[name] = result_dict

    # Comparison table
    if args.compare and len(all_results) > 1:
        print_comparison(all_results)

    # Save combined
    if all_results:
        combined_path = args.output_dir / "combined_results.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nCombined: {combined_path}")


def print_comparison(all_results: dict):
    """Print comparison table across parsers."""
    names = list(all_results.keys())

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")

    header = f"{'Metric':<30}" + "".join(f"{n:>15}" for n in names)
    print(header)
    print("-" * len(header))

    metrics = [
        ("Text NED (page_avg) ↓", "text", "ned_page_avg"),
        ("Text NED (edit_whole) ↓", "text", "ned_whole"),
        ("Table TEDS ↑", "table", "teds"),
        ("Table TEDS-S ↑", "table", "teds_s"),
        ("Formula NED ↓", "formula", "ned_avg"),
        ("Reading Order NED ↓", "reading_order", "ned_avg"),
    ]

    for label, cat, key in metrics:
        row = f"{label:<30}"
        for name in names:
            val = all_results[name].get(cat, {}).get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>15.4f}"
            else:
                row += f"{str(val):>15}"
        print(row)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
