"""Pseudo Ground Truth Generator.

Generates structured markdown GT from PDF pages using VLM (Vision-Language Model).
Converts each PDF page to an image, sends to VLM API, and assembles structured markdown.

Pipeline:
    PDF → page images (PyMuPDF) → VLM API (Qwen3-VL-30B) → Structured Markdown
    → page-level files + merged gt.md

Supports concurrent batch processing for faster throughput.
"""

import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF
import httpx

from training.prompts.templates import (
    PSEUDO_GT_SYSTEM_PROMPT,
    PSEUDO_GT_USER_PROMPT,
    PSEUDO_GT_SYSTEM_PROMPT_EN,
    PSEUDO_GT_USER_PROMPT_EN,
)


@dataclass
class PageResult:
    """Result of GT generation for a single page."""
    page_num: int
    markdown: str
    success: bool
    elapsed_time: float
    error: Optional[str] = None


@dataclass
class GTGenerationResult:
    """Result of GT generation for an entire document."""
    doc_id: str
    total_pages: int
    processed_pages: int
    failed_pages: list[int] = field(default_factory=list)
    total_time: float = 0.0
    output_path: Optional[str] = None


def pdf_page_to_base64(pdf_path: Path, page_num: int, dpi: int = 200) -> str:
    """Convert a single PDF page to base64-encoded PNG."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    png_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(png_bytes).decode("utf-8")


def call_vlm_with_image(
    image_b64: str,
    api_url: str,
    model: str,
    timeout: float = 180.0,
    max_tokens: int = 8192,
    lang: str = "ko",
) -> str:
    """Send a page image to VLM and get structured markdown back."""
    system_prompt = PSEUDO_GT_SYSTEM_PROMPT_EN if lang == "en" else PSEUDO_GT_SYSTEM_PROMPT
    user_prompt = PSEUDO_GT_USER_PROMPT_EN if lang == "en" else PSEUDO_GT_USER_PROMPT
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": True},
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()

    message = result["choices"][0]["message"]

    # 1차: reasoning-parser가 분리한 content 사용 (reasoning 필드가 있으면 분리 성공)
    if message.get("reasoning_content") is not None:
        raw = message["content"] or ""
    else:
        # 2차: reasoning-parser 미작동 시 raw content에서 후처리로 thinking 제거
        raw = message["content"] or ""

    return _clean_response(raw)


# Thinking 오염 패턴: 태그 없이 raw thinking이 출력된 경우 감지
_THINKING_PATTERNS = re.compile(
    r"^(Okay,? let'?s|Got it,? let'?s|First,? I need to|Let me (think|check|look|analyze|start))"
    r"|^Wait,? (the |let me|I need)"
    r"|^(But how|But the |However,? the |Perhaps the )"
    r"|^(Now,? (let'?s|I need|let me))"
    r"|^Looking at the (image|table|document)",
    re.IGNORECASE | re.MULTILINE,
)


def _clean_response(raw: str) -> str:
    """Remove thinking tags, raw thinking text, and code fences from VLM response.

    Handles three cases:
    1. Proper <think>...</think> tags → split on </think>, take content after
    2. Only </think> closing tag (template places <think> in prompt) → split on </think>
    3. No tags but raw thinking text → detect patterns and warn
    """
    content = raw

    # Case 1 & 2: </think> 태그가 있으면 그 이후만 사용
    if "</think>" in content:
        content = content.split("</think>", 1)[1]

    # <think> 열림 태그가 남아있으면 제거
    content = re.sub(r"<think>.*?(?:</think>|$)", "", content, flags=re.DOTALL)

    content = content.strip()

    # Code fence 제거
    if content.startswith("```markdown"):
        content = content[len("```markdown"):]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    # Case 3: 태그 없이 thinking이 본문에 섞인 경우 경고 로깅
    if content and _THINKING_PATTERNS.search(content[:500]):
        # thinking으로 시작하지만 실제 markdown 내용이 뒤에 있을 수 있음
        # 첫 markdown heading(#)이나 table(|)이 나오는 지점부터 사용
        md_match = re.search(r"^(#{1,6}\s|\|)", content, re.MULTILINE)
        if md_match and md_match.start() > 50:
            content = content[md_match.start():]

    return content


def _process_single_page(
    pdf_path: Path,
    page_num: int,
    page_file: Path,
    api_url: str,
    model: str,
    dpi: int,
    max_tokens: int,
    timeout: float,
    lang: str = "ko",
) -> PageResult:
    """Process a single page: render → VLM → save. Used by thread pool."""
    start = time.time()
    try:
        image_b64 = pdf_page_to_base64(pdf_path, page_num, dpi=dpi)
        markdown = call_vlm_with_image(
            image_b64=image_b64,
            api_url=api_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            lang=lang,
        )
        page_file.write_text(markdown, encoding="utf-8")
        elapsed = time.time() - start
        return PageResult(
            page_num=page_num,
            markdown=markdown,
            success=True,
            elapsed_time=elapsed,
        )
    except Exception as e:
        elapsed = time.time() - start
        return PageResult(
            page_num=page_num,
            markdown="",
            success=False,
            elapsed_time=elapsed,
            error=str(e),
        )


def generate_pseudo_gt(
    pdf_path: Path,
    output_dir: Path,
    api_url: str = "http://localhost:8000/v1/chat/completions",
    model: str = "qwen3-vl-30b-thinking",
    dpi: int = 200,
    max_tokens: int = 8192,
    timeout: float = 180.0,
    max_pages: Optional[int] = None,
    skip_existing: bool = True,
    batch_size: int = 4,
    lang: str = "ko",
) -> GTGenerationResult:
    """Generate pseudo GT for a PDF document.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save GT files
        api_url: VLM API endpoint
        model: VLM model name
        dpi: Image rendering DPI
        max_tokens: Max tokens per VLM response
        timeout: API request timeout (seconds)
        max_pages: Maximum pages to process (None = all)
        skip_existing: Skip if gt.md already exists
        batch_size: Number of concurrent VLM requests

    Returns:
        GTGenerationResult with generation stats
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = output_dir / "gt.md"
    pages_dir = output_dir / "gt_pages"

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    if max_pages:
        total_pages = min(total_pages, max_pages)

    doc_id = output_dir.name
    result = GTGenerationResult(doc_id=doc_id, total_pages=total_pages, processed_pages=0)

    if skip_existing and gt_path.exists():
        print(f"  [SKIP] {doc_id}: gt.md already exists")
        result.output_path = str(gt_path)
        result.processed_pages = total_pages
        return result

    pages_dir.mkdir(exist_ok=True)
    start_time = time.time()

    # Identify pages that need processing
    pages_to_process = []
    existing_pages = {}  # page_num -> markdown content

    for page_num in range(total_pages):
        page_file = pages_dir / f"page_{page_num + 1:04d}.md"
        if page_file.exists() and page_file.stat().st_size > 0:
            existing_pages[page_num] = page_file.read_text(encoding="utf-8")
            result.processed_pages += 1
        else:
            pages_to_process.append(page_num)

    skip_count = len(existing_pages)
    todo_count = len(pages_to_process)
    print(f"  [{doc_id}] {skip_count} cached, {todo_count} to process (batch={batch_size})")

    # Process pages concurrently in batches
    completed_pages = {}  # page_num -> markdown

    if pages_to_process:
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {}
            for page_num in pages_to_process:
                page_file = pages_dir / f"page_{page_num + 1:04d}.md"
                future = executor.submit(
                    _process_single_page,
                    pdf_path=pdf_path,
                    page_num=page_num,
                    page_file=page_file,
                    api_url=api_url,
                    model=model,
                    dpi=dpi,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    lang=lang,
                )
                futures[future] = page_num

            for future in as_completed(futures):
                page_num = futures[future]
                page_result = future.result()

                if page_result.success:
                    completed_pages[page_num] = page_result.markdown
                    result.processed_pages += 1
                    print(f"  [{doc_id}] page {page_num + 1}/{total_pages} "
                          f"OK ({page_result.elapsed_time:.1f}s, {len(page_result.markdown)} chars)")
                else:
                    result.failed_pages.append(page_num + 1)
                    completed_pages[page_num] = f"<!-- PAGE {page_num + 1}: GENERATION FAILED -->\n"
                    print(f"  [{doc_id}] page {page_num + 1}/{total_pages} "
                          f"FAIL ({page_result.elapsed_time:.1f}s) - {page_result.error}")

    # Merge all pages into gt.md (in order)
    merged = []
    for page_num in range(total_pages):
        if page_num > 0:
            merged.append("\n\n---\n\n")
        if page_num in existing_pages:
            merged.append(existing_pages[page_num])
        elif page_num in completed_pages:
            merged.append(completed_pages[page_num])
        else:
            merged.append(f"<!-- PAGE {page_num + 1}: MISSING -->\n")

    gt_path.write_text("".join(merged), encoding="utf-8")

    result.total_time = time.time() - start_time
    result.output_path = str(gt_path)

    # Save generation metadata
    meta = {
        "doc_id": doc_id,
        "model": model,
        "dpi": dpi,
        "batch_size": batch_size,
        "total_pages": total_pages,
        "processed_pages": result.processed_pages,
        "failed_pages": result.failed_pages,
        "total_time_seconds": round(result.total_time, 1),
        "avg_time_per_page": round(result.total_time / max(todo_count, 1), 1),
    }
    (output_dir / "gt_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return result
