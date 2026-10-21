#!/usr/bin/env python3
"""KoGovDoc MinerU baseline — run MinerU (magic-pdf) on val pages.

Uses magic-pdf with -s/-e for individual page extraction.
MinerU uses 0-indexed pages.

Usage:
    python scripts/eval_kogovdoc/run_mineru_baseline.py \
        --output-dir results/kogovdoc/mineru_val
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


def get_val_pages(val_jsonl: Path) -> dict[str, list[int]]:
    """Extract val page numbers per document from val.jsonl.
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
    pdf = datasets_dir / "documents" / doc_id / "doc.pdf"
    if pdf.exists():
        return pdf
    pdf = datasets_dir / "papers" / doc_id / "paper.pdf"
    if pdf.exists():
        return pdf
    return None


def run_mineru_page(pdf_path: Path, page_0idx: int, tmp_dir: Path) -> str:
    """Run magic-pdf on a single page. Returns markdown text."""
    cmd = [
        "magic-pdf",
        "-p", str(pdf_path),
        "-o", str(tmp_dir),
        "-m", "auto",
        "-s", str(page_0idx),
        "-e", str(page_0idx),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"MinerU failed: {result.stderr[:300]}")

    # Find the output .md file
    stem = pdf_path.stem
    md_file = tmp_dir / stem / "auto" / f"{stem}.md"
    if md_file.exists():
        return md_file.read_text(encoding="utf-8")

    raise RuntimeError(f"MinerU output not found at {md_file}")


def main():
    parser = argparse.ArgumentParser(description="MinerU baseline on KoGovDoc val")
    parser.add_argument("--val-jsonl", type=Path, default=Path("datasets/training/val.jsonl"))
    parser.add_argument("--datasets-dir", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/kogovdoc/mineru_val/predictions"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

        print(f"  [{doc_id}] {len(pages)} pages from {pdf_path.name}...", flush=True)

        for page_num in pages:
            out_file = args.output_dir / f"{doc_id}_page_{page_num:04d}.md"
            if out_file.exists():
                skipped += 1
                continue

            page_0idx = page_num - 1  # Convert to 0-indexed

            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    md_text = run_mineru_page(pdf_path, page_0idx, Path(tmp_dir))
                out_file.write_text(md_text, encoding="utf-8")
                done += 1
            except Exception as e:
                print(f"    page {page_num} ERROR: {e}")
                out_file.write_text(f"<!-- ERROR: {e} -->\n", encoding="utf-8")
                errors += 1

        elapsed = time.time() - start_time
        rate = (done + skipped + errors) / elapsed if elapsed > 0 else 0
        remaining = (total_pages - done - skipped - errors) / rate if rate > 0 else 0
        print(f"    {done+skipped} done, {errors} errors, ETA: {remaining/60:.0f}m")

    elapsed_total = time.time() - start_time
    print(f"\nDone in {elapsed_total/60:.1f}m")
    print(f"  Processed: {done}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")


if __name__ == "__main__":
    main()
