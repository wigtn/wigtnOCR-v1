#!/usr/bin/env python3
"""KoGovDoc BC/CS Evaluation — MoC-based chunking quality metrics.

Compares WigtnOCR (structured markdown) vs PaddleOCR (plain text)
across 3 chunking strategies: header-based, semantic, fixed-size.

Uses Qwen2.5-1.5B for perplexity-based BC/CS (MoC paper, ACL 2025).

Usage:
    python scripts/run_kogovdoc_bccs_eval.py \
        --ppl-api-url http://localhost:8020/v1 \
        --ppl-model qwen2.5-1.5b-instruct \
        --embedding-api-url http://localhost:8001/embeddings \
        --output-dir results/kogovdoc/bccs_eval
"""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics.chunking import (
    create_ppl_client,
    calculate_bc_from_fn,
    calculate_cs_from_fn,
    BCScore,
    CSScore,
)


# ---------------------------------------------------------------------------
# Chunking Strategies
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    content: str
    index: int


def chunk_header_based(text: str) -> list[Chunk]:
    """Split by markdown headings (# ## ### etc).

    Only works on structured markdown (VLM output).
    Falls back to single chunk if no headings found.
    """
    lines = text.split("\n")
    chunks = []
    current_lines = []

    for line in lines:
        if re.match(r"^#{1,6}\s", line.strip()) and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                chunks.append(Chunk(content=content, index=len(chunks)))
            current_lines = [line]
        else:
            current_lines.append(line)

    # Last chunk
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append(Chunk(content=content, index=len(chunks)))

    return chunks


def chunk_semantic(text: str, embedding_api_url: str, embedding_model: str) -> list[Chunk]:
    """Split by semantic similarity using BGE-M3 embeddings.

    Uses sentence-level embedding distance to find breakpoints.
    """
    import httpx

    # Split into sentences
    sentences = re.split(r'(?<=[.!?。！？\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if len(sentences) < 3:
        return [Chunk(content=text.strip(), index=0)] if text.strip() else []

    # Get embeddings for all sentences
    embeddings = []
    batch_size = 32
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        resp = httpx.post(
            embedding_api_url,
            json={"model": embedding_model, "input": batch},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        embeddings.extend([np.array(d["embedding"]) for d in data])

    # Calculate cosine distances between consecutive sentences
    distances = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        distances.append(1.0 - cos_sim)

    # Find breakpoints using percentile threshold
    if not distances:
        return [Chunk(content=text.strip(), index=0)]

    threshold = np.percentile(distances, 90)

    chunks = []
    current_sentences = [sentences[0]]

    for i, dist in enumerate(distances):
        if dist >= threshold and len(current_sentences) >= 2:
            content = " ".join(current_sentences).strip()
            if content:
                chunks.append(Chunk(content=content, index=len(chunks)))
            current_sentences = [sentences[i + 1]]
        else:
            current_sentences.append(sentences[i + 1])

    if current_sentences:
        content = " ".join(current_sentences).strip()
        if content:
            chunks.append(Chunk(content=content, index=len(chunks)))

    return chunks


def chunk_fixed_size(text: str, chunk_size: int = 512) -> list[Chunk]:
    """Split into fixed-size character chunks.

    Simple baseline — no intelligence, mechanical splitting.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    for i in range(0, len(text), chunk_size):
        content = text[i:i + chunk_size].strip()
        if content:
            chunks.append(Chunk(content=content, index=len(chunks)))

    return chunks


# ---------------------------------------------------------------------------
# BC/CS Evaluation for a single document
# ---------------------------------------------------------------------------

MAX_CHUNKS_FOR_CS = 50  # CS is O(n²) — cap to avoid hours-long computation


def _sample_chunks_evenly(contents: list[str], max_n: int) -> list[str]:
    """Sample chunks evenly across the document to keep representative coverage."""
    if len(contents) <= max_n:
        return contents
    indices = np.linspace(0, len(contents) - 1, max_n, dtype=int)
    return [contents[i] for i in indices]


def evaluate_single_document(
    chunks: list[Chunk],
    ppl_fn,
    doc_id: str,
    strategy: str,
    parser: str,
    skip_delta: int = 1,
    threshold_k: float = 0.1,
) -> dict:
    """Compute BC and CS for a list of chunks."""

    if len(chunks) < 2:
        return {
            "doc_id": doc_id,
            "parser": parser,
            "strategy": strategy,
            "n_chunks": len(chunks),
            "bc_mean": None,
            "cs": None,
            "cs_n_edges": 0,
            "status": "skipped_few_chunks",
        }

    contents = [c.content for c in chunks]

    # BC: adjacent pairs — O(n), always use full list
    bc_scores = []
    for i in range(len(contents) - 1):
        try:
            bc = calculate_bc_from_fn(contents[i], contents[i + 1], ppl_fn)
            bc_scores.append(bc.bc)
        except Exception as e:
            bc_scores.append(None)

    valid_bcs = [b for b in bc_scores if b is not None]
    bc_mean = sum(valid_bcs) / len(valid_bcs) if valid_bcs else None

    # CS: O(n²) — sample if too many chunks
    cs_sampled = len(contents) > MAX_CHUNKS_FOR_CS
    cs_contents = _sample_chunks_evenly(contents, MAX_CHUNKS_FOR_CS) if cs_sampled else contents

    try:
        cs_score, edge_weights = calculate_cs_from_fn(
            cs_contents, ppl_fn,
            threshold_k=threshold_k,
            skip_delta=skip_delta,
        )
        cs = cs_score.cs
        cs_n_edges = cs_score.n_edges
    except Exception as e:
        cs = None
        cs_n_edges = 0

    return {
        "doc_id": doc_id,
        "parser": parser,
        "strategy": strategy,
        "n_chunks": len(chunks),
        "n_chunks_cs": len(cs_contents) if cs_sampled else len(chunks),
        "cs_sampled": cs_sampled,
        "bc_mean": round(bc_mean, 4) if bc_mean is not None else None,
        "cs": round(cs, 4) if cs is not None else None,
        "cs_n_edges": cs_n_edges,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def aggregate_to_documents(pred_dir: Path) -> dict[str, str]:
    """Group page-level .md files into document-level text.

    Handles two cases:
    - Page-level files: kogov_004_page_0001.md → concatenated by doc
    - Document-level files: kogov_004_page_0001.md (single file per doc, e.g. Marker)
      Deduplicates identical content within same doc.
    """
    from collections import defaultdict

    files = sorted(pred_dir.glob("*.md"))
    doc_pages: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for f in files:
        parts = f.stem.split("_page_")
        if len(parts) == 2:
            doc_id = parts[0]
            page_key = parts[1]
        else:
            doc_id = f.stem
            page_key = "0000"
        doc_pages[doc_id].append((page_key, f))

    doc_texts = {}
    for doc_id, pages in doc_pages.items():
        pages.sort(key=lambda x: x[0])
        seen_texts = set()
        texts = []
        for _, page_file in pages:
            text = page_file.read_text(encoding="utf-8")
            if text.startswith("<!-- ERROR:"):
                continue
            text = text.strip()
            if text and len(text) > 10 and text not in seen_texts:
                texts.append(text)
                seen_texts.add(text)
        if texts:
            doc_texts[doc_id] = "\n\n".join(texts)
    return doc_texts


def main():
    parser = argparse.ArgumentParser(
        description="KoGovDoc BC/CS Evaluation (MoC framework)"
    )
    parser.add_argument(
        "--parsers", type=str, nargs="+",
        default=["wigtnocr", "paddleocr"],
        help="Parser names to evaluate (e.g., wigtnocr paddleocr marker mineru)",
    )
    parser.add_argument(
        "--pred-dirs", type=str, nargs="+",
        default=None,
        help="Prediction directories corresponding to --parsers. "
             "If not specified, uses default paths.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/kogovdoc/bccs_eval"),
    )
    parser.add_argument("--ppl-api-url", default="http://localhost:8020/v1")
    parser.add_argument("--ppl-model", default="qwen2.5-1.5b-instruct")
    parser.add_argument("--embedding-api-url", default="http://localhost:8001/embeddings")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--threshold-k", type=float, default=0.1, help="CS edge weight threshold K")
    parser.add_argument("--skip-delta", type=int, default=1, help="CS sequential skip constraint δ")
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=["header_based", "semantic", "fixed_size"],
        help="Chunking strategies to evaluate (e.g., semantic)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Default prediction directories per parser
    DEFAULT_PRED_DIRS = {
        "wigtnocr": "results/kogovdoc/v1_val/predictions",
        "paddleocr": "results/kogovdoc/paddleocr_val/predictions",
        "marker": "results/kogovdoc/marker_val/predictions",
        "mineru": "results/kogovdoc/mineru_val/predictions",
        "qwen3vl2b": "results/kogovdoc/2b_base_val/predictions",
        "qwen3vl30b": "results/kogovdoc/30b_val/predictions",
    }

    if args.pred_dirs:
        if len(args.pred_dirs) != len(args.parsers):
            print("ERROR: --pred-dirs must match --parsers count")
            sys.exit(1)
        pred_dirs = {name: Path(d) for name, d in zip(args.parsers, args.pred_dirs)}
    else:
        pred_dirs = {}
        for name in args.parsers:
            if name in DEFAULT_PRED_DIRS:
                pred_dirs[name] = Path(DEFAULT_PRED_DIRS[name])
            else:
                print(f"ERROR: No default pred-dir for parser '{name}'. Use --pred-dirs.")
                sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create ppl client
    print(f"PPL model: {args.ppl_model} @ {args.ppl_api_url}")
    ppl_fn = create_ppl_client(
        base_url=args.ppl_api_url,
        model=args.ppl_model,
    )

    # Load and aggregate predictions for each parser
    parsers_list = []
    for parser_name in args.parsers:
        pred_dir = pred_dirs[parser_name]
        if not pred_dir.exists():
            print(f"WARNING: {pred_dir} does not exist, skipping {parser_name}")
            continue
        doc_texts = aggregate_to_documents(pred_dir)
        n_files = len(list(pred_dir.glob("*.md")))
        print(f"{parser_name}: {n_files} files → {len(doc_texts)} documents")
        for doc_id in sorted(doc_texts):
            print(f"  {doc_id}: {len(doc_texts[doc_id]):,} chars")
        parsers_list.append((parser_name, doc_texts))

    # Find common document IDs across all parsers
    if not parsers_list:
        print("ERROR: No parsers loaded")
        sys.exit(1)

    common_ids = set(parsers_list[0][1].keys())
    for _, doc_texts in parsers_list[1:]:
        common_ids &= set(doc_texts.keys())
    common_ids = sorted(common_ids)

    print(f"\nCommon documents across all parsers: {len(common_ids)}")

    if args.limit:
        common_ids = common_ids[:args.limit]
        print(f"Limited to {args.limit} documents")

    # Resume support
    results_file = args.output_dir / "bccs_results.jsonl"
    done_keys = set()
    if args.resume and results_file.exists():
        with open(results_file, "r") as f:
            for line in f:
                r = json.loads(line)
                done_keys.add(f"{r['doc_id']}_{r['parser']}_{r['strategy']}")
        print(f"Resuming: {len(done_keys)} results already computed")

    strategies = args.strategies

    total_tasks = len(common_ids) * len(strategies) * len(parsers_list)
    total_tasks -= len(done_keys)
    parser_names = ", ".join(p[0] for p in parsers_list)
    print(f"\nTotal evaluations to run: {total_tasks}")
    print(f"Strategies: {strategies}")
    print(f"Parsers: {parser_names}")
    print(f"CS params: K={args.threshold_k}, δ={args.skip_delta}")
    print(f"{'='*65}\n")

    start_time = time.time()
    done = 0
    all_results = []

    with open(results_file, "a") as out_f:
        for doc_id in common_ids:
            for parser_name, doc_map in parsers_list:
                text = doc_map.get(doc_id, "")

                if not text.strip() or len(text.strip()) < 20:
                    continue

                for strategy in strategies:
                    key = f"{doc_id}_{parser_name}_{strategy}"
                    if key in done_keys:
                        continue

                    # Chunking
                    try:
                        if strategy == "header_based":
                            chunks = chunk_header_based(text)
                        elif strategy == "semantic":
                            chunks = chunk_semantic(
                                text, args.embedding_api_url, args.embedding_model
                            )
                        elif strategy == "fixed_size":
                            chunks = chunk_fixed_size(text, chunk_size=512)
                    except Exception as e:
                        print(f"  CHUNK ERROR: {doc_id}/{parser_name}/{strategy}: {e}")
                        continue

                    # BC/CS evaluation
                    try:
                        result = evaluate_single_document(
                            chunks=chunks,
                            ppl_fn=ppl_fn,
                            doc_id=doc_id,
                            strategy=strategy,
                            parser=parser_name,
                            skip_delta=args.skip_delta,
                            threshold_k=args.threshold_k,
                        )
                    except Exception as e:
                        print(f"  EVAL ERROR: {doc_id}/{parser_name}/{strategy}: {e}")
                        result = {
                            "doc_id": doc_id, "parser": parser_name,
                            "strategy": strategy, "status": f"error: {e}",
                            "n_chunks": len(chunks), "bc_mean": None, "cs": None,
                        }

                    # Write result
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    all_results.append(result)
                    done += 1

                    elapsed = time.time() - start_time
                    avg = elapsed / done if done > 0 else 0
                    remaining = (total_tasks - done) * avg

                    bc = result.get("bc_mean")
                    cs = result.get("cs")
                    bc_str = f"BC={bc:.4f}" if bc is not None else "BC=N/A"
                    cs_str = f"CS={cs:.4f}" if cs is not None else "CS=N/A"

                    print(
                        f"  [{done}/{total_tasks}] {doc_id} "
                        f"{parser_name:10s} {strategy:14s} "
                        f"chunks={result.get('n_chunks', 0):>3} "
                        f"{bc_str:>12} {cs_str:>12} "
                        f"ETA: {remaining/60:.0f}m",
                        flush=True,
                    )

    # Summary
    elapsed_total = time.time() - start_time
    print(f"\n{'='*65}")
    print(f"BC/CS Evaluation Complete — {elapsed_total/60:.1f} minutes")
    print(f"{'='*65}\n")

    # Aggregate by parser × strategy
    agg = {}
    for r in all_results:
        if r.get("status") != "ok":
            continue
        key = (r["parser"], r["strategy"])
        if key not in agg:
            agg[key] = {"bc_list": [], "cs_list": []}
        if r.get("bc_mean") is not None:
            agg[key]["bc_list"].append(r["bc_mean"])
        if r.get("cs") is not None:
            agg[key]["cs_list"].append(r["cs"])

    print(f"  {'Parser':<12} {'Strategy':<16} {'BC (avg)':>10} {'CS_i (avg)':>12} {'Samples':>8}")
    print(f"  {'-'*60}")

    summary = {}
    for (p, s), data in sorted(agg.items()):
        bc_avg = sum(data["bc_list"]) / len(data["bc_list"]) if data["bc_list"] else None
        cs_avg = sum(data["cs_list"]) / len(data["cs_list"]) if data["cs_list"] else None
        n = max(len(data["bc_list"]), len(data["cs_list"]))

        bc_str = f"{bc_avg:.4f}" if bc_avg is not None else "N/A"
        cs_str = f"{cs_avg:.4f}" if cs_avg is not None else "N/A"
        print(f"  {p:<12} {s:<16} {bc_str:>10} {cs_str:>12} {n:>8}")

        summary[f"{p}_{s}"] = {
            "bc_avg": round(bc_avg, 4) if bc_avg is not None else None,
            "cs_avg": round(cs_avg, 4) if cs_avg is not None else None,
            "n_samples": n,
        }

    print(f"\n  BC: higher = better (chunks are independent)")
    print(f"  CS: lower = better (less stickiness)")
    print(f"{'='*65}")

    # Save summary
    summary_path = args.output_dir / "bccs_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")
    print(f"Detail saved: {results_file}")


if __name__ == "__main__":
    main()
