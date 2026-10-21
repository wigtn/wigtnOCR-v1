#!/usr/bin/env python3
"""KoGovDoc PaddleOCR baseline — extract plain text from val images.

Uses RapidOCR (PaddleOCR ONNX runtime) to extract text from KoGovDoc
validation images. Produces plain text without any structural markup,
serving as the "unstructured OCR baseline" for BC/CS comparison.

Usage:
    python scripts/run_kogovdoc_paddleocr_baseline.py \
        --output-dir results/kogovdoc/paddleocr_val
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from rapidocr_onnxruntime import RapidOCR


def ocr_single_image(image_path: str) -> str:
    """Run RapidOCR on a single image, return plain text."""
    ocr = RapidOCR()
    result, _ = ocr(image_path)
    if not result:
        return ""
    # Sort by vertical position (top of bounding box) for reading order
    sorted_lines = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
    return "\n".join(line[1] for line in sorted_lines)


def process_sample(args_tuple) -> dict:
    """Process a single sample: OCR image → save text."""
    sample_id, image_path, output_path = args_tuple

    # Skip if already done
    if output_path.exists() and output_path.stat().st_size > 0:
        content = output_path.read_text(encoding="utf-8")
        if not content.startswith("<!-- ERROR:"):
            return {"status": "skipped", "id": sample_id}

    start = time.time()
    try:
        text = ocr_single_image(str(image_path))
        if not text:
            raise RuntimeError("Empty OCR result")
        output_path.write_text(text, encoding="utf-8")
        elapsed = time.time() - start
        return {
            "status": "ok",
            "id": sample_id,
            "elapsed": round(elapsed, 1),
            "chars": len(text),
        }
    except Exception as e:
        elapsed = time.time() - start
        output_path.write_text(f"<!-- ERROR: {e} -->", encoding="utf-8")
        return {
            "status": "error",
            "id": sample_id,
            "elapsed": round(elapsed, 1),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="KoGovDoc PaddleOCR baseline (RapidOCR)"
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("datasets/training/val.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kogovdoc/paddleocr_val"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load val.jsonl
    samples = []
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} validation samples")

    if args.limit:
        samples = samples[: args.limit]

    # Parse samples
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for sample in samples:
        image_path = Path(sample["images"][0])
        parts = image_path.parts
        sample_id = f"{parts[-2]}_{parts[-1].replace('.png', '').replace('.jpg', '')}"
        output_path = pred_dir / f"{sample_id}.md"
        tasks.append((sample_id, image_path, output_path))

    # Filter already done
    to_process = []
    for t in tasks:
        sample_id, _, output_path = t
        if output_path.exists() and output_path.stat().st_size > 0:
            content = output_path.read_text(encoding="utf-8")
            if not content.startswith("<!-- ERROR:"):
                continue
        to_process.append(t)

    skipped = len(tasks) - len(to_process)
    print(f"To process: {len(to_process)} (skipping {skipped} already done)")

    if not to_process:
        print("All samples already processed.")
        return

    start_time = time.time()
    counts = {"ok": 0, "error": 0, "skipped": 0}
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_sample, t): t[0] for t in to_process
        }

        for future in as_completed(futures):
            result = future.result()
            done += 1
            counts[result["status"]] = counts.get(result["status"], 0) + 1

            elapsed = time.time() - start_time
            avg = elapsed / done
            remaining = (len(to_process) - done) * avg

            if result["status"] == "ok":
                print(
                    f"  [{done}/{len(to_process)}] {result['id']} "
                    f"OK ({result['elapsed']}s, {result['chars']} chars) "
                    f"ETA: {remaining/60:.0f}m",
                    flush=True,
                )
            elif result["status"] == "error":
                print(
                    f"  [{done}/{len(to_process)}] {result['id']} "
                    f"ERROR: {result.get('error', 'unknown')}",
                    flush=True,
                )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"PaddleOCR baseline complete")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  OK: {counts.get('ok', 0)}")
    print(f"  Errors: {counts.get('error', 0)}")
    print(f"  Skipped: {counts.get('skipped', 0)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
