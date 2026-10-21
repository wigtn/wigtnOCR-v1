#!/usr/bin/env python3
"""KoGovDoc Marker baseline — run Marker PDF parser on val pages.

Uses marker_single with --page_range to process only val pages.
Marker uses 0-indexed pages, our dataset uses 1-indexed (page_0001 = page 1).

Usage:
    python scripts/eval_kogovdoc/run_marker_baseline.py \
        --output-dir results/kogovdoc/marker_val
"""

import argparse
import ast
import json
import subprocess
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path


def get_val_pages(val_jsonl: Path) -> dict[str, list[int]]:
    """Extract val page numbers per document from val.jsonl.

    Returns dict like {'kogov_003': [1, 11, 24, ...], 'arxiv_004': [9, 35, ...]}
    Page numbers are 1-indexed (matching filename convention).
    """
    pages_per_doc = defaultdict(list)
    with open(val_jsonl) as f:
        for line in f:
            r = json.loads(line)
            imgs = r["images"] if isinstance(r["images"], list) else json.loads(r["images"])
            for img in imgs:
                parts = img.split("/")
                doc_id = parts[-2]
                page_num = int(parts[-1].replace("page_", "").replace(".png", ""))
                pages_per_doc[doc_id].append(page_num)

    return {k: sorted(set(v)) for k, v in pages_per_doc.items()}


def find_pdf(doc_id: str, datasets_dir: Path) -> Path | None:
    """Find the source PDF for a document."""
    # KoGovDoc: datasets/documents/kogov_XXX/doc.pdf
    pdf = datasets_dir / "documents" / doc_id / "doc.pdf"
    if pdf.exists():
        return pdf
    # ArXiv: datasets/papers/arxiv_XXX/paper.pdf
    pdf = datasets_dir / "papers" / doc_id / "paper.pdf"
    if pdf.exists():
        return pdf
    return None


def run_marker_single(
    pdf_path: Path,
    page_range: str,
    output_dir: Path,
) -> str:
    """Run marker_single on a PDF with specific page range.

    Returns the markdown output text.
    """
    # marker_single saves to a default location, we'll capture it
    cmd = [
        "marker_single",
        str(pdf_path),
        "--page_range", page_range,
        "--output_format", "markdown",
        "--disable_image_extraction",
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Marker failed: {result.stderr[:500]}")

    # Find the output — marker saves to conversion_results/<stem>/
    # The path is printed in the log
    for line in result.stderr.split("\n"):
        if "Saved markdown to" in line:
            saved_dir = line.split("Saved markdown to ")[-1].strip()
            md_file = Path(saved_dir) / f"{pdf_path.stem}.md"
            if md_file.exists():
                text = md_file.read_text(encoding="utf-8")
                # Cleanup marker output dir
                shutil.rmtree(Path(saved_dir), ignore_errors=True)
                return text

    raise RuntimeError(f"Could not find Marker output for {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Marker baseline on KoGovDoc val")
    parser.add_argument(
        "--val-jsonl", type=Path,
        default=Path("datasets/training/val.jsonl"),
    )
    parser.add_argument(
        "--datasets-dir", type=Path,
        default=Path("datasets"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/kogovdoc/marker_val/predictions"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get val pages
    val_pages = get_val_pages(args.val_jsonl)
    total_pages = sum(len(v) for v in val_pages.values())
    print(f"Val pages: {total_pages} across {len(val_pages)} documents")

    start_time = time.time()
    done = 0
    errors = 0
    skipped = 0

    for doc_id in sorted(val_pages.keys()):
        pages = val_pages[doc_id]
        pdf_path = find_pdf(doc_id, args.datasets_dir)

        if pdf_path is None:
            print(f"  [{doc_id}] PDF not found, skipping {len(pages)} pages")
            errors += len(pages)
            continue

        # Convert 1-indexed pages to 0-indexed for Marker
        page_range_str = ",".join(str(p - 1) for p in pages)

        print(f"  [{doc_id}] {len(pages)} pages from {pdf_path.name}...", end=" ", flush=True)

        try:
            md_text = run_marker_single(pdf_path, page_range_str, args.output_dir)
        except Exception as e:
            print(f"ERROR: {e}")
            # Save error markers for each page
            for page_num in pages:
                out_file = args.output_dir / f"{doc_id}_page_{page_num:04d}.md"
                if not out_file.exists():
                    out_file.write_text(f"<!-- ERROR: {e} -->\n", encoding="utf-8")
            errors += len(pages)
            continue

        # Marker outputs all pages concatenated — split by page markers if possible
        # For now, save the whole output per page (Marker doesn't add page breaks)
        # We'll use the concatenated text for document-level BC/CS anyway
        #
        # Strategy: split by "---" or page-like markers, or just save whole doc
        # Since BC/CS is document-level, save one file per document page
        # For individual page comparison, we'd need page-level splitting

        # Simple approach: split text roughly by page count
        # But Marker doesn't guarantee page boundaries...
        # Save whole document output, and individual pages get the full text
        # This is fine for document-level BC/CS evaluation

        # Save per-page files (each page gets its portion)
        # Since marker processes specific pages, the output IS those pages
        lines = md_text.split("\n")
        if len(pages) == 1:
            # Single page — straightforward
            out_file = args.output_dir / f"{doc_id}_page_{pages[0]:04d}.md"
            if not out_file.exists():
                out_file.write_text(md_text, encoding="utf-8")
                done += 1
            else:
                skipped += 1
        else:
            # Multiple pages — save concatenated output per doc
            # Individual page files get the full concat (for doc-level aggregation)
            for page_num in pages:
                out_file = args.output_dir / f"{doc_id}_page_{page_num:04d}.md"
                if not out_file.exists():
                    out_file.write_text(md_text, encoding="utf-8")
                    done += 1
                else:
                    skipped += 1

        elapsed = time.time() - start_time
        print(f"OK ({len(pages)} pages, {elapsed:.0f}s total)")

    elapsed_total = time.time() - start_time
    print(f"\nDone in {elapsed_total:.0f}s")
    print(f"  Processed: {done}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")


if __name__ == "__main__":
    main()
