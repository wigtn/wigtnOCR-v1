"""GT Quality Validator.

Validates pseudo GT quality using a larger LLM (122B) as text-based judge.
Sends generated GT markdown to the judge model, which evaluates structural quality,
formatting consistency, and signs of AI contamination (thinking tags, hallucination).

Pipeline:
    GT markdown → 122B Judge LLM → ValidationResult (JSON)
    → per-page scores + document-level aggregation → validation_report.json
"""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import httpx

from training.gt_generator import _clean_response
from training.prompts.templates import GT_VALIDATION_SYSTEM_PROMPT, GT_VALIDATION_USER_PROMPT


@dataclass
class ValidationResult:
    """Result of GT validation for a single page."""
    page_num: int
    score: int  # 1-5 overall
    structure_quality: int = 0
    table_quality: int = 0
    completeness_signals: int = 0
    hallucination_signals: int = 0
    formatting_consistency: int = 0
    errors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    is_acceptable: bool = True  # score >= 3
    elapsed_time: float = 0.0
    error: Optional[str] = None  # API/parse error


@dataclass
class DocumentValidationResult:
    """Aggregated validation result for a document."""
    doc_id: str
    total_pages: int
    sampled_pages: int
    avg_score: float = 0.0
    min_score: int = 5
    max_score: int = 1
    acceptable_ratio: float = 0.0
    failed_pages: list[int] = field(default_factory=list)  # score < 3
    api_errors: list[int] = field(default_factory=list)  # API call failures
    page_results: list[ValidationResult] = field(default_factory=list)
    total_time: float = 0.0


def _parse_validation_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling common formatting issues."""
    content = _clean_response(raw)
    if not content.startswith("{"):
        start = content.find("{")
        if start != -1:
            content = content[start:]
    if not content.endswith("}"):
        end = content.rfind("}")
        if end != -1:
            content = content[:end + 1]
    return json.loads(content)


def validate_single_page(
    gt_markdown: str,
    api_url: str,
    model: str,
    timeout: float = 180.0,
    max_tokens: int = 16384,
) -> dict:
    """Send GT markdown to judge LLM and get validation scores."""
    user_content = GT_VALIDATION_USER_PROMPT.format(gt_markdown=gt_markdown)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": GT_VALIDATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()

    raw = result["choices"][0]["message"]["content"]
    if raw is None:
        raise ValueError("Model returned empty content (thinking budget may have exceeded max_tokens)")
    return _parse_validation_json(raw)


def _validate_page_worker(
    page_num: int,
    gt_markdown: str,
    api_url: str,
    model: str,
    timeout: float,
) -> ValidationResult:
    """Worker function for thread pool: validate a single page."""
    start = time.time()
    try:
        scores = validate_single_page(
            gt_markdown=gt_markdown,
            api_url=api_url,
            model=model,
            timeout=timeout,
        )
        elapsed = time.time() - start
        overall = scores.get("score", 3)
        return ValidationResult(
            page_num=page_num,
            score=overall,
            structure_quality=scores.get("structure_quality", 0),
            table_quality=scores.get("table_quality", 0),
            completeness_signals=scores.get("completeness_signals", 0),
            hallucination_signals=scores.get("hallucination_signals", 0),
            formatting_consistency=scores.get("formatting_consistency", 0),
            errors=scores.get("errors", []),
            suggestions=scores.get("suggestions", []),
            is_acceptable=overall >= 3,
            elapsed_time=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - start
        return ValidationResult(
            page_num=page_num,
            score=0,
            is_acceptable=False,
            elapsed_time=elapsed,
            error=str(e),
        )


def validate_document(
    doc_dir: Path,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "qwen3.5-122b",
    timeout: float = 180.0,
    sample_ratio: float = 0.1,
    batch_size: int = 2,
    seed: int = 42,
) -> DocumentValidationResult:
    """Validate pseudo GT quality for a document by sampling pages.

    Args:
        doc_dir: Document directory containing gt_pages/
        api_url: Judge LLM API endpoint
        model: Judge LLM model name
        timeout: API request timeout (seconds)
        sample_ratio: Fraction of pages to sample (0.0-1.0)
        batch_size: Concurrent validation requests
        seed: Random seed for reproducible sampling
    """
    doc_id = doc_dir.name
    pages_dir = doc_dir / "gt_pages"

    # Count available pages with GT content
    available_pages = []
    page_files = sorted(pages_dir.glob("page_*.md"))
    for page_file in page_files:
        if page_file.stat().st_size > 10:
            # Extract page number from filename (page_0001.md -> 0)
            page_num = int(page_file.stem.split("_")[1]) - 1
            available_pages.append(page_num)

    total_pages = len(page_files)

    if not available_pages:
        return DocumentValidationResult(
            doc_id=doc_id, total_pages=total_pages, sampled_pages=0,
        )

    # Sample pages
    sample_count = max(1, int(len(available_pages) * sample_ratio))
    rng = random.Random(seed)
    sampled = sorted(rng.sample(available_pages, min(sample_count, len(available_pages))))

    result = DocumentValidationResult(
        doc_id=doc_id, total_pages=total_pages, sampled_pages=len(sampled),
    )

    print(f"  [{doc_id}] {len(available_pages)} pages with GT, "
          f"sampling {len(sampled)} ({sample_ratio:.0%}), batch={batch_size}")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {}
        for page_num in sampled:
            page_file = pages_dir / f"page_{page_num + 1:04d}.md"
            gt_markdown = page_file.read_text(encoding="utf-8")
            future = executor.submit(
                _validate_page_worker,
                page_num=page_num,
                gt_markdown=gt_markdown,
                api_url=api_url,
                model=model,
                timeout=timeout,
            )
            futures[future] = page_num

        for future in as_completed(futures):
            page_num = futures[future]
            vr = future.result()
            result.page_results.append(vr)

            if vr.error:
                result.api_errors.append(page_num + 1)
                print(f"  [{doc_id}] page {page_num + 1} "
                      f"ERROR ({vr.elapsed_time:.1f}s) - {vr.error}")
            else:
                status = "PASS" if vr.is_acceptable else "FAIL"
                print(f"  [{doc_id}] page {page_num + 1} "
                      f"{status} score={vr.score}/5 ({vr.elapsed_time:.1f}s)"
                      f"{' errors: ' + str(vr.errors) if vr.errors else ''}")
                if not vr.is_acceptable:
                    result.failed_pages.append(page_num + 1)

    # Sort results by page number
    result.page_results.sort(key=lambda r: r.page_num)

    # Compute aggregates (exclude API errors)
    valid_results = [r for r in result.page_results if r.error is None]
    if valid_results:
        scores = [r.score for r in valid_results]
        result.avg_score = sum(scores) / len(scores)
        result.min_score = min(scores)
        result.max_score = max(scores)
        result.acceptable_ratio = sum(1 for r in valid_results if r.is_acceptable) / len(valid_results)

    result.total_time = time.time() - start_time
    return result


def result_to_dict(result: DocumentValidationResult) -> dict:
    """Convert DocumentValidationResult to serializable dict."""
    return {
        "doc_id": result.doc_id,
        "total_pages": result.total_pages,
        "sampled_pages": result.sampled_pages,
        "avg_score": round(result.avg_score, 2),
        "min_score": result.min_score,
        "max_score": result.max_score,
        "acceptable_ratio": round(result.acceptable_ratio, 3),
        "failed_pages": result.failed_pages,
        "api_errors": result.api_errors,
        "total_time_seconds": round(result.total_time, 1),
        "page_results": [
            {
                "page": r.page_num + 1,
                "score": r.score,
                "structure_quality": r.structure_quality,
                "table_quality": r.table_quality,
                "completeness_signals": r.completeness_signals,
                "hallucination_signals": r.hallucination_signals,
                "formatting_consistency": r.formatting_consistency,
                "is_acceptable": r.is_acceptable,
                "errors": r.errors,
                "suggestions": r.suggestions,
                "elapsed_time": round(r.elapsed_time, 1),
                "api_error": r.error,
            }
            for r in result.page_results
        ],
    }
