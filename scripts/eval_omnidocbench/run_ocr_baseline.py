#!/usr/bin/env python3
"""Run OSS OCR baselines on OmniDocBench dataset.

CPU-based OCR/document parsers that can run in parallel without GPU.

Supported parsers:
  - marker: Marker PDF→Markdown (layout + surya OCR) [OmniDocBench official]
  - mineru: MinerU PDF parser (layout + OCR) [OmniDocBench official]
  - docling: IBM Docling (layout-aware document parsing)
  - tesseract: Tesseract OCR (text extraction only)

Usage:
    # Marker (OmniDocBench baseline)
    python scripts/run_omnidocbench_ocr_baseline.py \
        --parser marker \
        --output-dir results/omnidocbench/marker \
        --workers 4

    # MinerU (OmniDocBench baseline)
    python scripts/run_omnidocbench_ocr_baseline.py \
        --parser mineru \
        --output-dir results/omnidocbench/mineru \
        --workers 4

    # Docling (CPU, parallel)
    python scripts/run_omnidocbench_ocr_baseline.py \
        --parser docling \
        --output-dir results/omnidocbench/docling \
        --workers 8

    # Tesseract (CPU, parallel)
    python scripts/run_omnidocbench_ocr_baseline.py \
        --parser tesseract \
        --output-dir results/omnidocbench/tesseract \
        --workers 16
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def parse_with_marker(image_path: str) -> str:
    """Parse document image using Marker (surya OCR + layout detection)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    import fitz
    import tempfile
    import os

    # Convert image to single-page PDF via PyMuPDF
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    with fitz.open(stream=img_bytes) as doc:
        pdf_bytes = doc.convert_to_pdf()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_pdf = tmp.name
        tmp.write(pdf_bytes)

    try:
        converter = PdfConverter(artifact_dict=create_model_dict())
        result = converter(tmp_pdf)
        return result.markdown
    finally:
        os.unlink(tmp_pdf)


def parse_with_mineru(image_path: str) -> str:
    """Parse document image using MinerU (magic-pdf)."""
    from magic_pdf.tools.common import do_parse
    import fitz
    import tempfile
    import os
    import glob

    # Convert image to single-page PDF via PyMuPDF (same as MinerU CLI)
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    with fitz.open(stream=img_bytes) as doc:
        pdf_bytes = doc.convert_to_pdf()

    stem = Path(image_path).stem
    with tempfile.TemporaryDirectory() as tmp_dir:
        do_parse(
            tmp_dir,
            stem,
            pdf_bytes,
            [],
            "auto",
            debug_able=False,
            f_draw_span_bbox=False,
            f_draw_layout_bbox=False,
            f_dump_md=True,
            f_dump_middle_json=False,
            f_dump_model_json=False,
            f_dump_orig_pdf=False,
            f_dump_content_list=False,
        )
        # Find the output .md file
        md_files = glob.glob(os.path.join(tmp_dir, "**", "*.md"), recursive=True)
        if md_files:
            with open(md_files[0], "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""


def parse_with_pymupdf4llm(image_path: str) -> str:
    """Parse document image using pymupdf4llm (PyMuPDF layout + Tesseract OCR)."""
    import pymupdf4llm
    import fitz
    import tempfile
    import os

    # Convert image to single-page PDF
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    with fitz.open(stream=img_bytes) as doc:
        pdf_bytes = doc.convert_to_pdf()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_pdf = tmp.name
        tmp.write(pdf_bytes)

    try:
        md = pymupdf4llm.to_markdown(
            tmp_pdf,
            force_ocr=True,
            ocr_language="eng+chi_sim+kor",
        )
        return md.strip()
    finally:
        os.unlink(tmp_pdf)


def parse_with_docling(image_path: str) -> str:
    """Parse document image using IBM Docling."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(image_path)
    return result.document.export_to_markdown()


def parse_with_tesseract(image_path: str) -> str:
    """Parse document image using Tesseract OCR."""
    import pytesseract
    from PIL import Image

    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="eng+chi_sim+kor")
    return text.strip()


PARSERS = {
    "marker": parse_with_marker,
    "mineru": parse_with_mineru,
    "pymupdf4llm": parse_with_pymupdf4llm,
    "docling": parse_with_docling,
    "tesseract": parse_with_tesseract,
}


def process_single(args_tuple: tuple) -> dict:
    """Process a single image (for ProcessPoolExecutor)."""
    image_path_str, output_path_str, parser_name = args_tuple
    image_path = Path(image_path_str)
    output_path = Path(output_path_str)

    # Skip if already processed
    if output_path.exists() and output_path.stat().st_size > 0:
        return {
            "image": image_path.name,
            "status": "skipped",
            "elapsed": 0,
            "chars": output_path.stat().st_size,
        }

    start = time.time()
    try:
        parse_fn = PARSERS[parser_name]
        markdown = parse_fn(image_path_str)
        if not markdown:
            markdown = ""
        output_path.write_text(markdown, encoding="utf-8")
        elapsed = time.time() - start
        return {
            "image": image_path.name,
            "status": "ok",
            "elapsed": round(elapsed, 1),
            "chars": len(markdown),
        }
    except Exception as e:
        elapsed = time.time() - start
        output_path.write_text(f"<!-- ERROR: {e} -->", encoding="utf-8")
        return {
            "image": image_path.name,
            "status": "error",
            "elapsed": round(elapsed, 1),
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Run OSS OCR baselines on OmniDocBench")
    parser.add_argument(
        "--parser",
        choices=list(PARSERS.keys()),
        required=True,
        help="OCR parser to use",
    )
    parser.add_argument(
        "--omnidocbench-dir",
        type=Path,
        default=Path("datasets/omnidocbench"),
        help="Path to OmniDocBench dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for prediction .md files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (CPU-based, can go high)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (for testing)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-process images that previously failed",
    )

    args = parser.parse_args()

    # Validate
    images_dir = args.omnidocbench_dir / "images"
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    # Collect images
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    print(f"Found {len(image_files)} images in {images_dir}")

    if args.limit:
        image_files = image_files[:args.limit]
        print(f"Limited to {args.limit} images")

    # Handle retry
    if args.retry_failed and args.output_dir.exists():
        removed = 0
        for md_file in args.output_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            if content.startswith("<!-- ERROR:"):
                md_file.unlink()
                removed += 1
        if removed:
            print(f"Removed {removed} failed files for retry")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Filter already processed
    existing = set(p.stem for p in args.output_dir.glob("*.md"))
    to_process = [p for p in image_files if p.stem not in existing]
    skipped = len(image_files) - len(to_process)
    print(f"To process: {len(to_process)} (skipping {skipped} already done)")

    if not to_process:
        print("All images already processed.")
        return

    # Build task list
    tasks = [
        (str(img), str(args.output_dir / f"{img.stem}.md"), args.parser)
        for img in to_process
    ]

    # Run with ProcessPoolExecutor (CPU parallelism)
    start_time = time.time()
    results = {"ok": 0, "error": 0, "skipped": skipped}
    done_count = 0

    print(f"Starting {args.parser} with {args.workers} workers...")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, t): t for t in tasks}

        for future in as_completed(futures):
            result = future.result()
            done_count += 1
            status = result["status"]
            results[status] = results.get(status, 0) + 1

            elapsed = time.time() - start_time
            avg = elapsed / done_count
            remaining = (len(to_process) - done_count) * avg

            if done_count % 50 == 0 or status == "error":
                if status == "ok":
                    print(
                        f"  [{done_count}/{len(to_process)}] "
                        f"OK ({result['elapsed']}s, {result['chars']} chars) "
                        f"ETA: {remaining/60:.0f}m"
                    )
                elif status == "error":
                    print(
                        f"  [{done_count}/{len(to_process)}] "
                        f"ERROR: {result.get('error', 'unknown')}"
                    )

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"OCR baseline complete: {args.parser}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Workers: {args.workers}")
    print(f"  OK: {results.get('ok', 0)}")
    print(f"  Errors: {results.get('error', 0)}")
    print(f"  Skipped: {results.get('skipped', 0)}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    # Save metadata
    meta = {
        "parser": args.parser,
        "workers": args.workers,
        "total_images": len(image_files),
        "processed": results.get("ok", 0),
        "errors": results.get("error", 0),
        "skipped": results.get("skipped", 0),
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_image": round(total_time / max(done_count, 1), 1),
    }
    meta_path = args.output_dir / "inference_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
