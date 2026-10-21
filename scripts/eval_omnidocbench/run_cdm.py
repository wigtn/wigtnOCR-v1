#!/usr/bin/env python3
"""Extract formula pairs from OmniDocBench predictions and run CDM evaluation.

Workflow:
  1. Extract matched formula pairs (gt LaTeX, pred LaTeX) from evaluator
  2. Export to CDM input JSON format
  3. Run CDM evaluation (LaTeX → image rendering → character matching)
  4. Merge CDM scores back into results JSON

Usage:
    python scripts/run_omnidocbench_cdm.py \
        --pred-dir results/omnidocbench/vlm_30b \
        --name vlm_30b \
        --pools 8

    # Run all models at once
    python scripts/run_omnidocbench_cdm.py --all --pools 8
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.omnidocbench.evaluator import OmniDocBenchEvaluator
from evaluation.omnidocbench.extractor import extract_gt_elements, extract_elements_from_markdown
from evaluation.omnidocbench.matcher import match_formula_elements


CDM_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "metrics" / "cdm"
OMNIDOCBENCH_JSON = Path("datasets/omnidocbench/OmniDocBench.json")
TINYTEX_BIN = Path.home() / ".TinyTeX" / "bin" / "x86_64-linux"


def extract_formula_pairs(pred_dir: Path) -> list[dict]:
    """Extract matched (gt, pred) formula LaTeX pairs."""
    with open(OMNIDOCBENCH_JSON, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    pairs = []
    skipped = 0

    for page_data in gt_data:
        page_info = page_data.get("page_info", {
            "image_path": page_data.get("image_path", ""),
        })
        image_path = page_info.get("image_path", page_data.get("image_path", ""))
        image_name = Path(image_path).stem if image_path else ""

        # Find prediction file
        pred_path = pred_dir / f"{image_name}.md"
        if not pred_path.exists():
            skipped += 1
            continue

        pred_content = pred_path.read_text(encoding="utf-8")
        if not pred_content or pred_content.startswith("<!-- ERROR:"):
            skipped += 1
            continue

        # Extract elements
        gt_elements = extract_gt_elements(page_data)
        pred_elements = extract_elements_from_markdown(pred_content)

        # Match formulas
        formula_matches = match_formula_elements(
            gt_elements["formula"], pred_elements["formula"],
        )

        for m in formula_matches:
            if m.get("matched") and m.get("gt") and m.get("pred"):
                gt_latex = m["gt"].strip()
                pred_latex = m["pred"].strip()
                if gt_latex and pred_latex:
                    pairs.append({
                        "img_id": f"{image_name}_f{len(pairs)}",
                        "gt": gt_latex,
                        "pred": pred_latex,
                    })

    print(f"  Extracted {len(pairs)} formula pairs (skipped {skipped} pages)")
    return pairs


def run_cdm(input_json: Path, output_dir: Path, pools: int = 8) -> dict:
    """Run CDM evaluation and return results."""
    # Set PATH to include TinyTeX
    env = os.environ.copy()
    tinytex_paths = [
        str(TINYTEX_BIN),
        str(Path.home() / ".local" / "bin"),
    ]
    env["PATH"] = ":".join(tinytex_paths) + ":" + env.get("PATH", "")

    cmd = [
        sys.executable,
        str(CDM_DIR / "evaluation.py"),
        "-i", str(input_json),
        "-o", str(output_dir),
        "-p", str(pools),
    ]

    print(f"  Running CDM with {pools} pools...")
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode != 0:
        print(f"  CDM stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"CDM evaluation failed (exit code {result.returncode})")

    # Parse output for mean score
    for line in result.stdout.split("\n"):
        if "mean f1 score" in line:
            print(f"  {line.strip()}")

    # Read metrics
    input_name = input_json.stem
    metrics_path = output_dir / input_name / "metrics_res.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def merge_cdm_results(results_json: Path, cdm_metrics: dict):
    """Merge CDM scores into existing results JSON."""
    if not results_json.exists():
        print(f"  Warning: {results_json} not found, creating new")
        results = {}
    else:
        with open(results_json) as f:
            results = json.load(f)

    results["formula"] = {
        "cdm_f1": cdm_metrics.get("mean_score", 0.0),
        "cdm_exp_rate": cdm_metrics.get("exp_rate", 0.0),
        "n_samples": len(cdm_metrics.get("details", {})),
        # Keep NED for reference
        **{k: v for k, v in results.get("formula", {}).items()
           if k.startswith("ned")},
    }

    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"  Updated: {results_json}")


def process_model(name: str, pred_dir: Path, pools: int):
    """Full pipeline for one model."""
    print(f"\n{'='*60}")
    print(f"  CDM Evaluation: {name}")
    print(f"{'='*60}")

    # Step 1: Extract formula pairs
    pairs = extract_formula_pairs(pred_dir)
    if not pairs:
        print("  No formula pairs found. Skipping.")
        return

    # Step 2: Write CDM input JSON
    cdm_input_dir = Path("results/omnidocbench/cdm_inputs")
    cdm_input_dir.mkdir(parents=True, exist_ok=True)
    input_json = cdm_input_dir / f"{name}.json"
    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"  Wrote CDM input: {input_json} ({len(pairs)} pairs)")

    # Step 3: Run CDM
    cdm_output_dir = Path("results/omnidocbench/cdm_output")
    cdm_output_dir.mkdir(parents=True, exist_ok=True)
    cdm_metrics = run_cdm(input_json, cdm_output_dir, pools)

    if cdm_metrics:
        print(f"  CDM F1: {cdm_metrics.get('mean_score', 'N/A')}")
        print(f"  CDM ExpRate: {cdm_metrics.get('exp_rate', 'N/A')}")

        # Step 4: Merge into results
        results_json = Path(f"results/omnidocbench/{name}_results.json")
        merge_cdm_results(results_json, cdm_metrics)
    else:
        print("  CDM evaluation returned no results")


def main():
    parser = argparse.ArgumentParser(description="Run CDM formula evaluation on OmniDocBench predictions")
    parser.add_argument("--pred-dir", type=Path, help="Prediction directory")
    parser.add_argument("--name", type=str, help="Model name")
    parser.add_argument("--pools", type=int, default=8, help="CDM parallel pools")
    parser.add_argument("--all", action="store_true", help="Run all models")
    args = parser.parse_args()

    if args.all:
        models = [
            ("vlm_30b", Path("results/omnidocbench/vlm_30b")),
            ("vlm_2b", Path("results/omnidocbench/vlm_2b")),
            ("marker", Path("results/omnidocbench/marker")),
        ]
        for name, pred_dir in models:
            if pred_dir.exists():
                process_model(name, pred_dir, args.pools)
            else:
                print(f"  Skipping {name}: {pred_dir} not found")
    else:
        if not args.pred_dir or not args.name:
            parser.error("--pred-dir and --name required (or use --all)")
        process_model(args.name, args.pred_dir, args.pools)

    # Update combined results
    combined = {}
    for name in ["vlm_30b", "vlm_2b", "marker"]:
        rpath = Path(f"results/omnidocbench/{name}_results.json")
        if rpath.exists():
            with open(rpath) as f:
                combined[name] = json.load(f)

    combined_path = Path("results/omnidocbench/combined_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
