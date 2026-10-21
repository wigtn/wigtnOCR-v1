"""Training data preparation for ms-swift VLM fine-tuning.

Converts PDF pages + GT markdown into ms-swift JSONL format:
    {"messages": [...], "images": ["/path/to/page.png"]}

Pipeline:
    1. Load validation report → filter pages with score >= min_score
    2. PDF pages → PNG images (PyMuPDF)
    3. GT markdown (per-page) → assistant response
    4. Downsample overrepresented documents (kogov_008 bias correction)
    5. Train/val split → train.jsonl + val.jsonl
"""

import json
import random
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from training.prompts.templates import (
    VLM_TRAINING_SYSTEM_PROMPT,
    STUDENT_USER_PROMPT,
    STUDENT_USER_PROMPT_EN,
)


def _render_page_image(pdf_path: Path, page_num: int, output_path: Path, dpi: int = 200):
    """Render a single PDF page to PNG."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(str(output_path))
    doc.close()


def _load_valid_pages(dataset_dir: Path, min_score: int = 3) -> dict[str, list[int]]:
    """Load validation report and return pages with score >= min_score per document.

    Returns:
        dict mapping doc_id -> list of 1-based page numbers that passed validation.
        Documents without a validation report include all pages with GT.
    """
    report_path = dataset_dir / "validation_report.json"

    # Collect validated page scores
    validated = {}  # doc_id -> {page_num -> score}
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        for doc in report["documents"]:
            page_scores = {}
            for pr in doc["page_results"]:
                if pr.get("api_error") is None:
                    page_scores[pr["page"]] = pr["score"]
            validated[doc["doc_id"]] = page_scores

    # For each document, determine which pages to include
    result = {}
    for doc_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        doc_id = doc_dir.name
        pages_dir = doc_dir / "gt_pages"
        if not pages_dir.exists():
            continue

        page_files = sorted(pages_dir.glob("page_*.md"))
        if not page_files:
            continue

        valid_pages = []
        for page_file in page_files:
            if page_file.stat().st_size <= 10:
                continue
            page_num = int(page_file.stem.split("_")[1])  # 1-based

            if doc_id in validated:
                # Only include if validated and passed
                score = validated[doc_id].get(page_num)
                if score is not None and score >= min_score:
                    valid_pages.append(page_num)
                elif score is None:
                    # Not sampled for validation — include by default
                    valid_pages.append(page_num)
            else:
                # No validation report — include all
                valid_pages.append(page_num)

        if valid_pages:
            result[doc_id] = sorted(valid_pages)

    return result


def _downsample_documents(
    pages_map: dict[str, list[int]],
    max_ratio: float = 0.25,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Downsample documents that exceed max_ratio of total pages.

    Prevents single large documents (e.g. kogov_008) from dominating training.
    """
    total = sum(len(pages) for pages in pages_map.values())
    max_pages = int(total * max_ratio)

    rng = random.Random(seed)
    result = {}
    for doc_id, pages in pages_map.items():
        if len(pages) > max_pages:
            sampled = sorted(rng.sample(pages, max_pages))
            print(f"  [{doc_id}] downsampled: {len(pages)} -> {len(sampled)} pages "
                  f"(max_ratio={max_ratio:.0%})")
            result[doc_id] = sampled
        else:
            result[doc_id] = pages

    return result


def prepare_training_data(
    dataset_dirs: list[Path],
    output_dir: Path,
    dpi: int = 200,
    min_score: int = 3,
    max_doc_ratio: float = 0.25,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """Prepare training data from multiple dataset directories.

    Args:
        dataset_dirs: List of dataset directories (e.g. [datasets/documents, datasets/papers])
        output_dir: Output directory for images and JSONL files
        dpi: Image rendering DPI
        min_score: Minimum validation score to include (1-5)
        max_doc_ratio: Maximum fraction any single document can represent
        val_ratio: Fraction of data for validation split
        seed: Random seed for reproducibility

    Returns:
        Summary statistics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    all_samples = []

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        lang = "en" if dataset_name == "papers" else "ko"
        user_prompt = STUDENT_USER_PROMPT_EN if lang == "en" else STUDENT_USER_PROMPT

        print(f"\n--- Processing {dataset_name} ---")

        # Load valid pages
        pages_map = _load_valid_pages(dataset_dir, min_score=min_score)
        if not pages_map:
            print(f"  No valid pages found for {dataset_name}")
            continue

        total_before = sum(len(p) for p in pages_map.values())
        print(f"  Valid pages (score>={min_score}): {total_before}")

        # Downsample overrepresented documents
        pages_map = _downsample_documents(pages_map, max_ratio=max_doc_ratio, seed=seed)
        total_after = sum(len(p) for p in pages_map.values())
        print(f"  After downsampling: {total_after}")

        # Generate samples
        for doc_id, pages in sorted(pages_map.items()):
            doc_dir = dataset_dir / doc_id
            pdf_path = doc_dir / "doc.pdf"
            if not pdf_path.exists():
                pdf_path = doc_dir / "paper.pdf"
            if not pdf_path.exists():
                print(f"  [{doc_id}] SKIP — no PDF")
                continue

            pages_dir = doc_dir / "gt_pages"
            doc_images_dir = images_dir / dataset_name / doc_id
            doc_images_dir.mkdir(parents=True, exist_ok=True)

            for page_num in pages:
                # GT markdown
                gt_file = pages_dir / f"page_{page_num:04d}.md"
                if not gt_file.exists() or gt_file.stat().st_size <= 10:
                    continue
                gt_markdown = gt_file.read_text(encoding="utf-8").strip()
                if not gt_markdown:
                    continue

                # Render page image
                image_path = doc_images_dir / f"page_{page_num:04d}.png"
                if not image_path.exists():
                    try:
                        _render_page_image(pdf_path, page_num - 1, image_path, dpi=dpi)
                    except Exception as e:
                        print(f"  [{doc_id}] page {page_num} render FAIL: {e}")
                        continue

                # Build ms-swift training sample
                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": VLM_TRAINING_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": f"<image>{user_prompt}",
                        },
                        {
                            "role": "assistant",
                            "content": gt_markdown,
                        },
                    ],
                    "images": [str(image_path.resolve())],
                }
                all_samples.append(sample)

        print(f"  [{dataset_name}] Generated {len(all_samples)} samples so far")

    if not all_samples:
        print("No samples generated!")
        return {"total": 0}

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(all_samples)

    val_count = max(1, int(len(all_samples) * val_ratio))
    val_samples = all_samples[:val_count]
    train_samples = all_samples[val_count:]

    # Write JSONL files
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    for path, samples in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    stats = {
        "total_samples": len(all_samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "images_dir": str(images_dir),
        "dpi": dpi,
        "min_score": min_score,
        "max_doc_ratio": max_doc_ratio,
        "seed": seed,
    }

    # Save stats
    stats_path = output_dir / "data_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 50}")
    print(f"Data preparation complete")
    print(f"  Train: {len(train_samples)} samples -> {train_path}")
    print(f"  Val:   {len(val_samples)} samples -> {val_path}")
    print(f"  Images: {images_dir}")
    print(f"{'=' * 50}")

    return stats
