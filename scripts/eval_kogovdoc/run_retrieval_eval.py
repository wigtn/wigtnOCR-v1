#!/usr/bin/env python3
"""KoGovDoc Retrieval Evaluation — Stage 8.

For each parser, semantic-chunk the document-level text, embed chunks with BGE-M3,
build a FAISS index, auto-generate queries from GT, and measure Hit@K, MRR, nDCG.

This proves: better chunks → better retrieval.

Usage:
    python scripts/eval_kogovdoc/run_retrieval_eval.py \
        --parsers wigtnocr paddleocr marker mineru qwen3vl2b qwen3vl30b \
        --embedding-api-url http://localhost:8001/embeddings \
        --output-dir results/kogovdoc/retrieval_eval
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.metrics.retrieval import evaluate_retrieval, RetrievalMetrics


# ---------------------------------------------------------------------------
# Embedding client
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], api_url: str, model: str, batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts using the embedding API. Returns (n, dim) array."""
    import httpx

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = httpx.post(
            api_url,
            json={"model": model, "input": batch},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        all_embeddings.extend([d["embedding"] for d in data])
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Semantic chunking (same as BC/CS script)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    content: str
    index: int
    doc_id: str = ""


def chunk_semantic(text: str, embedding_api_url: str, embedding_model: str) -> list[Chunk]:
    """Split by semantic similarity using embeddings."""
    import httpx

    sentences = re.split(r'(?<=[.!?。！？\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if len(sentences) < 3:
        return [Chunk(content=text.strip(), index=0)] if text.strip() else []

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

    distances = []
    for i in range(len(embeddings) - 1):
        a, b = embeddings[i], embeddings[i + 1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        distances.append(1.0 - cos_sim)

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


# ---------------------------------------------------------------------------
# Query generation from GT
# ---------------------------------------------------------------------------

def generate_queries_from_gt(gt_dir: Path, doc_ids: list[str]) -> dict[str, list[str]]:
    """Generate queries from GT text by extracting key sentences.

    For each document, extract heading-like lines and first sentences of sections
    as pseudo-queries. Returns {doc_id: [query1, query2, ...]}.
    """
    queries = {}
    for doc_id in doc_ids:
        # Find GT files for this doc
        gt_files = sorted(gt_dir.glob(f"{doc_id}_page_*.md"))
        if not gt_files:
            # Try looking in dataset GT
            continue

        full_text = ""
        for f in gt_files:
            text = f.read_text(encoding="utf-8").strip()
            if text and not text.startswith("<!-- ERROR"):
                full_text += "\n" + text

        if not full_text.strip():
            continue

        doc_queries = []
        lines = full_text.split("\n")
        for line in lines:
            line = line.strip()
            # Extract headings as queries
            if re.match(r'^#{1,3}\s+.+', line):
                heading = re.sub(r'^#+\s+', '', line).strip()
                if len(heading) > 5:
                    doc_queries.append(heading)
            # Extract first sentence of paragraphs (non-heading, substantial)
            elif len(line) > 30 and not line.startswith("|") and not line.startswith("```"):
                # Take first sentence
                first_sent = re.split(r'[.!?。！？]', line)[0].strip()
                if len(first_sent) > 15:
                    doc_queries.append(first_sent)

        # Limit queries per doc
        if doc_queries:
            # Deduplicate and limit
            seen = set()
            unique = []
            for q in doc_queries:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            queries[doc_id] = unique[:20]  # Max 20 queries per doc

    return queries


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate_to_documents(pred_dir: Path) -> dict[str, str]:
    """Group page-level .md files into document-level text."""
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="KoGovDoc Retrieval Evaluation")
    parser.add_argument(
        "--parsers", type=str, nargs="+",
        default=["wigtnocr", "paddleocr", "marker", "mineru", "qwen3vl2b", "qwen3vl30b"],
    )
    parser.add_argument("--embedding-api-url", default="http://localhost:8001/embeddings")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/kogovdoc/retrieval_eval"),
    )
    parser.add_argument(
        "--gt-dir", type=Path,
        default=Path("results/kogovdoc/v1_val/predictions"),
        help="GT directory for query generation (use best parser as reference)",
    )
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    DEFAULT_PRED_DIRS = {
        "wigtnocr": "results/kogovdoc/v1_val/predictions",
        "paddleocr": "results/kogovdoc/paddleocr_val/predictions",
        "marker": "results/kogovdoc/marker_val/predictions",
        "mineru": "results/kogovdoc/mineru_val/predictions",
        "qwen3vl2b": "results/kogovdoc/2b_base_val/predictions",
        "qwen3vl30b": "results/kogovdoc/30b_val/predictions",
    }

    # Load all parsers
    parsers_data = {}
    for name in args.parsers:
        pred_dir = Path(DEFAULT_PRED_DIRS[name])
        if not pred_dir.exists():
            print(f"WARNING: {pred_dir} not found, skipping {name}")
            continue
        doc_texts = aggregate_to_documents(pred_dir)
        print(f"{name}: {len(doc_texts)} documents")
        parsers_data[name] = doc_texts

    # Find common documents
    common_ids = set(list(parsers_data.values())[0].keys())
    for doc_texts in parsers_data.values():
        common_ids &= set(doc_texts.keys())
    common_ids = sorted(common_ids)
    print(f"Common documents: {len(common_ids)}")

    # Generate queries from GT (use wigtnocr predictions as reference for query generation)
    print(f"\nGenerating queries from GT ({args.gt_dir})...")
    queries = generate_queries_from_gt(args.gt_dir, common_ids)
    total_queries = sum(len(v) for v in queries.values())
    print(f"Generated {total_queries} queries across {len(queries)} documents")

    if total_queries == 0:
        print("ERROR: No queries generated. Check GT directory.")
        sys.exit(1)

    # For each parser: chunk → embed → index → retrieve → evaluate
    all_results = {}
    start_time = time.time()

    for parser_name, doc_texts in parsers_data.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {parser_name}")
        print(f"{'='*60}")

        # 1. Semantic chunk all documents
        print("  Chunking...", end=" ", flush=True)
        all_chunks = []  # (chunk_id, content, doc_id)
        for doc_id in common_ids:
            text = doc_texts.get(doc_id, "")
            if not text or len(text.strip()) < 20:
                continue
            try:
                chunks = chunk_semantic(text, args.embedding_api_url, args.embedding_model)
                for chunk in chunks:
                    chunk_id = f"{doc_id}_chunk_{chunk.index}"
                    all_chunks.append((chunk_id, chunk.content, doc_id))
            except Exception as e:
                print(f"\n  CHUNK ERROR {doc_id}: {e}")
                continue
        print(f"{len(all_chunks)} chunks")

        if len(all_chunks) < 2:
            print(f"  Skipping {parser_name} — too few chunks")
            continue

        # 2. Embed all chunks
        print("  Embedding chunks...", end=" ", flush=True)
        chunk_texts = [c[1] for c in all_chunks]
        chunk_embeddings = embed_texts(chunk_texts, args.embedding_api_url, args.embedding_model)
        print(f"shape {chunk_embeddings.shape}")

        # 3. Build FAISS index
        print("  Building FAISS index...", end=" ", flush=True)
        import faiss
        dim = chunk_embeddings.shape[1]
        # Normalize for cosine similarity
        faiss.normalize_L2(chunk_embeddings)
        index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
        index.add(chunk_embeddings)
        print(f"done ({index.ntotal} vectors)")

        # 4. Retrieve for each query
        print("  Retrieving...", end=" ", flush=True)
        max_k = max(args.k_values)
        qrels = {}   # {query_id: {chunk_id: relevance}}
        results = {} # {query_id: {chunk_id: score}}

        query_count = 0
        for doc_id, doc_queries in queries.items():
            # Find chunks belonging to this document
            doc_chunk_ids = [c[0] for c in all_chunks if c[2] == doc_id]
            if not doc_chunk_ids:
                continue

            for qi, query in enumerate(doc_queries):
                query_id = f"{doc_id}_q{qi}"

                # qrels: chunks from the same document are relevant
                qrels[query_id] = {cid: 1 for cid in doc_chunk_ids}

                # Embed query and search
                query_emb = embed_texts([query], args.embedding_api_url, args.embedding_model)
                faiss.normalize_L2(query_emb)
                scores, indices = index.search(query_emb, min(max_k, index.ntotal))

                results[query_id] = {}
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        results[query_id][all_chunks[idx][0]] = float(score)

                query_count += 1

        print(f"{query_count} queries evaluated")

        # 5. Compute metrics
        metrics = evaluate_retrieval(qrels, results, k_values=args.k_values)
        flat = metrics.flat_dict()
        all_results[parser_name] = flat

        print(f"\n  Results for {parser_name}:")
        for k in args.k_values:
            print(f"    Hit@{k}: {flat[f'Hit@{k}']:.4f}  "
                  f"MRR@{k}: {flat[f'MRR@{k}']:.4f}  "
                  f"nDCG@{k}: {flat[f'nDCG@{k}']:.4f}")

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Retrieval Evaluation Complete — {elapsed/60:.1f} minutes")
    print(f"{'='*60}\n")

    # Summary table
    print(f"  {'Parser':<12}", end="")
    for k in args.k_values:
        print(f" {'Hit@'+str(k):>8} {'MRR@'+str(k):>8} {'nDCG@'+str(k):>9}", end="")
    print()
    print(f"  {'-'*(12 + len(args.k_values) * 27)}")

    for parser_name, flat in all_results.items():
        print(f"  {parser_name:<12}", end="")
        for k in args.k_values:
            h = flat.get(f"Hit@{k}", 0)
            m = flat.get(f"MRR@{k}", 0)
            n = flat.get(f"nDCG@{k}", 0)
            print(f" {h:>8.4f} {m:>8.4f} {n:>9.4f}", end="")
        print()

    # Save results
    results_path = args.output_dir / "retrieval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
