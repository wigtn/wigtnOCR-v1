#!/usr/bin/env python3
"""KoGovDoc validation set evaluation.

Reads val.jsonl (294 samples), sends each page image to the VLM API,
and compares predicted markdown against GT (assistant response in val.jsonl).

Usage:
    python scripts/run_kogovdoc_val_eval.py \
        --model wigtnocr-2b-v1 \
        --api-url http://localhost:8010/v1/chat/completions \
        --output-dir results/kogovdoc/v1_val \
        --batch-size 8
"""

import argparse
import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics.ned import calculate_ned


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def call_vlm(
    image_b64: str,
    api_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
    timeout: float = 180.0,
) -> str:
    """Send page image to VLM API and get markdown."""
    import httpx

    suffix = "png" if image_b64[:8].startswith("iVBOR") else "jpeg"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{suffix};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()

    raw = result["choices"][0]["message"]["content"] or ""
    return _clean_response(raw)


_HEADING_RE = re.compile(r"^#{1,6}\s")
_MARKDOWN_ANCHOR_RE = re.compile(
    r"^(#{1,6}\s|\||\$\$|- |\* |\d+\.\s|<table|!\[|>|```|\\\[|\[Figure)",
)
_REASONING_RE = re.compile(
    r"^(Okay,? |Got it,? |Let me |Wait,? |Hmm,? |Actually,? |"
    r"I need to |I should |I think |I'll |Let's |"
    r"Looking at the (image|document|table|page)|"
    r"Now,? (let me|I need|I'll)|"
    r"So,? (the final|let me|I need)|"
    r"Alright,? )",
    re.IGNORECASE,
)


def _find_content_start(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if _HEADING_RE.match(line.strip()):
            return i
    return None


def _clean_response(raw: str) -> str:
    lines = raw.split("\n")
    start = _find_content_start(lines)
    if start and start > 0:
        lines = lines[start:]

    kept_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue
        if stripped.startswith("```"):
            if not kept_lines:
                continue
            if not any(_MARKDOWN_ANCHOR_RE.match(l.strip()) for l in kept_lines if l.strip()):
                continue
            kept_lines.append(line)
            continue
        if _REASONING_RE.match(stripped):
            continue
        kept_lines.append(line)

    result = "\n".join(kept_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Normalize for NED
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Process single sample
# ---------------------------------------------------------------------------

def process_sample(
    sample_id: str,
    image_path: Path,
    system_prompt: str,
    user_prompt: str,
    output_path: Path,
    api_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> dict:
    """Process a single val sample."""
    # Skip if already done
    if output_path.exists() and output_path.stat().st_size > 0:
        content = output_path.read_text(encoding="utf-8")
        if not content.startswith("<!-- ERROR:"):
            return {"status": "skipped", "id": sample_id, "elapsed": 0}

    start = time.time()
    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        markdown = call_vlm(
            image_b64=image_b64,
            api_url=api_url,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if not markdown:
            raise RuntimeError("Empty response after cleanup")
        output_path.write_text(markdown, encoding="utf-8")
        elapsed = time.time() - start
        return {
            "status": "ok",
            "id": sample_id,
            "elapsed": round(elapsed, 1),
            "chars": len(markdown),
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KoGovDoc validation set evaluation"
    )
    parser.add_argument(
        "--val-jsonl",
        type=Path,
        default=Path("datasets/training/val.jsonl"),
        help="Path to val.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8010/v1/chat/completions",
        help="VLM API endpoint",
    )
    parser.add_argument(
        "--model",
        default="wigtnocr-2b-v1",
        help="VLM model name",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of concurrent VLM requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip inference, only evaluate existing predictions",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Evaluation name (default: model name)",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Folder holding documents/ and papers/ page images, e.g. the images/ folder of "
             "Wigtn/KoGovDoc-Bench. Replaces the machine-specific directory in val.jsonl paths.",
    )

    args = parser.parse_args()
    eval_name = args.name or args.model
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load val.jsonl
    samples = []
    with open(args.val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} validation samples")

    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {args.limit} samples")

    # Parse samples: extract image path, prompts, GT
    parsed = []
    for i, sample in enumerate(samples):
        image_path = Path(sample["images"][0])
        if args.image_root:
            image_path = args.image_root.joinpath(*image_path.parts[-3:])
        messages = sample["messages"]

        system_prompt = ""
        user_prompt = ""
        gt_text = ""

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                # Extract text part (skip <image> tag)
                content = msg["content"]
                if isinstance(content, str):
                    user_prompt = content.replace("<image>", "").strip()
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            user_prompt = part["text"]
            elif msg["role"] == "assistant":
                gt_text = msg["content"]

        # Generate sample ID from image path
        # e.g., documents/kogov_008/page_0544.png → kogov_008_page_0544
        parts = image_path.parts
        sample_id = f"{parts[-2]}_{parts[-1].replace('.png', '').replace('.jpg', '')}"

        parsed.append({
            "id": sample_id,
            "image_path": image_path,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "gt_text": gt_text,
        })

    # Inference
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        # Handle retry-failed
        if args.retry_failed:
            removed = 0
            for p in parsed:
                pred_file = pred_dir / f"{p['id']}.md"
                if pred_file.exists():
                    content = pred_file.read_text(encoding="utf-8")
                    if content.startswith("<!-- ERROR:"):
                        pred_file.unlink()
                        removed += 1
            if removed:
                print(f"Removed {removed} failed files for retry")

        # Filter already done
        to_process = []
        for p in parsed:
            pred_file = pred_dir / f"{p['id']}.md"
            if pred_file.exists() and pred_file.stat().st_size > 0:
                content = pred_file.read_text(encoding="utf-8")
                if not content.startswith("<!-- ERROR:"):
                    continue
            to_process.append(p)

        skipped = len(parsed) - len(to_process)
        print(f"To process: {len(to_process)} (skipping {skipped} already done)")

        if to_process:
            start_time = time.time()
            counts = {"ok": 0, "error": 0}
            done = 0

            with ThreadPoolExecutor(max_workers=args.batch_size) as executor:
                futures = {}
                for p in to_process:
                    pred_file = pred_dir / f"{p['id']}.md"
                    future = executor.submit(
                        process_sample,
                        sample_id=p["id"],
                        image_path=p["image_path"],
                        system_prompt=p["system_prompt"],
                        user_prompt=p["user_prompt"],
                        output_path=pred_file,
                        api_url=args.api_url,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )
                    futures[future] = p["id"]

                for future in as_completed(futures):
                    result = future.result()
                    done += 1
                    status = result["status"]
                    counts[status] = counts.get(status, 0) + 1

                    elapsed = time.time() - start_time
                    avg = elapsed / done
                    remaining = (len(to_process) - done) * avg

                    if status == "ok":
                        print(
                            f"  [{done}/{len(to_process)}] {result['id']} "
                            f"OK ({result['elapsed']}s, {result['chars']} chars) "
                            f"ETA: {remaining/60:.0f}m",
                            flush=True,
                        )
                    elif status == "error":
                        print(
                            f"  [{done}/{len(to_process)}] {result['id']} "
                            f"ERROR: {result.get('error', 'unknown')}",
                            flush=True,
                        )

            total_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Inference complete: {args.model}")
            print(f"  Total time: {total_time/60:.1f} minutes")
            print(f"  OK: {counts.get('ok', 0)}")
            print(f"  Errors: {counts.get('error', 0)}")
            print(f"{'='*60}", flush=True)

    # Evaluation
    print(f"\nRunning evaluation...", flush=True)
    all_neds = []
    per_doc = {}
    error_count = 0
    skip_count = 0

    for p in parsed:
        pred_file = pred_dir / f"{p['id']}.md"

        if not pred_file.exists():
            skip_count += 1
            continue

        pred_text = pred_file.read_text(encoding="utf-8")
        if pred_text.startswith("<!-- ERROR:"):
            error_count += 1
            continue

        gt_text = p["gt_text"]

        pred_norm = normalize_text(pred_text)
        gt_norm = normalize_text(gt_text)
        ned = calculate_ned(pred_norm, gt_norm)
        all_neds.append(ned)

        # Per-document aggregation
        doc_id = p["id"].rsplit("_page_", 1)[0]  # kogov_008
        if doc_id not in per_doc:
            per_doc[doc_id] = []
        per_doc[doc_id].append(ned)

    # Summary
    ned_avg = sum(all_neds) / len(all_neds) if all_neds else None

    print(f"\n{'='*65}")
    print(f"  KoGovDoc Val Evaluation: {eval_name}")
    print(f"{'='*65}")
    print(f"  Val samples:     {len(parsed)}")
    print(f"  Evaluated:       {len(all_neds)}")
    print(f"  Errors:          {error_count}")
    print(f"  Skipped:         {skip_count}")
    if ned_avg is not None:
        print(f"  NED (avg):       {ned_avg:.4f}  (lower=better)")
    print()

    if per_doc:
        print(f"  {'Document':<15} {'NED avg':>10} {'Pages':>8}")
        print(f"  {'-'*33}")
        for doc_id in sorted(per_doc.keys()):
            neds = per_doc[doc_id]
            avg = sum(neds) / len(neds)
            print(f"  {doc_id:<15} {avg:>10.4f} {len(neds):>8}")
    print(f"{'='*65}", flush=True)

    # Save results
    results = {
        "name": eval_name,
        "total_samples": len(parsed),
        "evaluated": len(all_neds),
        "errors": error_count,
        "skipped": skip_count,
        "ned_avg": round(ned_avg, 4) if ned_avg else None,
        "per_doc": {
            doc_id: {
                "ned_avg": round(sum(neds) / len(neds), 4),
                "pages": len(neds),
            }
            for doc_id, neds in sorted(per_doc.items())
        },
    }

    result_path = args.output_dir / "eval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved: {result_path}")

    summary_dir = Path("results/kogovdoc")
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{eval_name}_val_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
