#!/usr/bin/env python3
"""
Chunking Evaluation CLI

MoC-based chunking quality evaluation for VLM document parsing.
Evaluates chunking quality using BC (Boundary Clarity) and CS (Chunk Stickiness)
metrics that don't require Ground Truth labels.

Reference: MoC Paper (arXiv:2503.09600v2)

Usage:
    # Full evaluation (all test folders in results/)
    python -m src.eval_chunking --all

    # With existing parsed files
    python -m src.eval_chunking --parsed-dir results/parsing/test_1/

    # Custom breakpoint settings
    python -m src.eval_chunking --all --breakpoint-type percentile --breakpoint-threshold 90
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# =============================================================================
# Import Compatibility Layer
# =============================================================================
# 두 가지 실행 방식 모두 지원:
#   1. python -m src.eval_chunking (프로젝트 루트에서)
#   2. python eval_chunking.py (src/ 디렉토리에서)

try:
    # 방법 1: src.xxx (프로젝트 루트에서 실행)
    from src.chunking.chunker import (
        ChunkerConfig,
        create_chunker,
        Chunk,
    )
    from src.chunking.metrics import (
        EmbeddingClient,
        APIEmbeddingClient,
        MockEmbeddingClient,
        create_embedding_client,
        evaluate_chunking,
        ChunkingMetrics,
    )
    from src.eval_parsers import FileFormat, detect_file_format
except ImportError:
    # 방법 2: xxx (src/ 디렉토리에서 실행 또는 PYTHONPATH 설정)
    from chunking.chunker import (
        ChunkerConfig,
        create_chunker,
        Chunk,
    )
    from chunking.metrics import (
        EmbeddingClient,
        APIEmbeddingClient,
        MockEmbeddingClient,
        create_embedding_client,
        evaluate_chunking,
        ChunkingMetrics,
    )
    from eval_parsers import FileFormat, detect_file_format


# =============================================================================
# Chunking Pipeline
# =============================================================================

def chunk_text(
    text: str,
    breakpoint_type: str = "percentile",
    breakpoint_threshold: float = 95.0,
    min_chunk_size: Optional[int] = None,
    embedding_api_url: str = "http://localhost:8001/embeddings",
    embedding_model: str = "BAAI/bge-m3",
    document_id: str = "doc"
) -> list[Chunk]:
    """Chunk text using semantic chunking.

    Args:
        text: Input text to chunk
        breakpoint_type: Type of breakpoint detection (percentile, standard_deviation, interquartile, gradient)
        breakpoint_threshold: Threshold amount for breakpoint detection
        min_chunk_size: Minimum chunk size (optional)
        embedding_api_url: API URL for embeddings
        embedding_model: Model name for embeddings
        document_id: Document identifier for chunk IDs

    Returns:
        List of Chunk objects
    """
    config = ChunkerConfig(
        breakpoint_threshold_type=breakpoint_type,
        breakpoint_threshold_amount=breakpoint_threshold,
        min_chunk_size=min_chunk_size,
    )

    chunker = create_chunker(
        config=config,
        embedding_api_url=embedding_api_url,
        embedding_model=embedding_model,
    )
    return chunker.chunk(text, document_id)


def load_parsed_files(
    parsed_dir: Path,
    breakpoint_type: str = "percentile",
    breakpoint_threshold: float = 95.0,
    min_chunk_size: Optional[int] = None,
    embedding_api_url: str = "http://localhost:8001/embeddings",
    embedding_model: str = "BAAI/bge-m3",
) -> dict[str, list[Chunk]]:
    """Load pre-parsed files from a directory and chunk them.

    Expected file naming:
    - vlm_output.md or vlm_output.txt
    - ocr-text_output.md or ocr-text_output.txt
    - ocr-image_output.md or ocr-image_output.txt
    - twostage-text_output.md or twostage-text_output.txt
    - twostage-image_output.md or twostage-image_output.txt

    Args:
        parsed_dir: Directory containing parsed output files
        breakpoint_type: Type of breakpoint detection
        breakpoint_threshold: Threshold amount for breakpoint detection
        min_chunk_size: Minimum chunk size (optional)
        embedding_api_url: API URL for embeddings
        embedding_model: Model name for embeddings

    Returns:
        Dictionary mapping parser name to list of chunks
    """
    results = {}

    # Map filename patterns to parser names
    # Support both old and new naming conventions
    patterns = {
        # New naming convention
        "image_advanced": "Image-Advanced",
        "image_baseline": "Image-Baseline",
        "text_advanced": "Text-Advanced",
        "text_baseline": "Text-Baseline",
        # Old naming convention (for compatibility)
        "vlm": "VLM",
        "ocr-text": "OCR-Text",
        "ocr-image": "OCR-Image",
        "twostage-text": "TwoStage-Text",
        "twostage-image": "TwoStage-Image",
    }

    for pattern, parser_name in patterns.items():
        # Try different extensions
        for ext in [".md", ".txt"]:
            file_path = parsed_dir / f"{pattern}_output{ext}"
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                chunks = chunk_text(
                    content,
                    breakpoint_type=breakpoint_type,
                    breakpoint_threshold=breakpoint_threshold,
                    min_chunk_size=min_chunk_size,
                    embedding_api_url=embedding_api_url,
                    embedding_model=embedding_model,
                    document_id=pattern,
                )
                results[parser_name] = chunks
                print(f"Loaded {parser_name}: {len(chunks)} chunks from {file_path.name}")
                break

    return results


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_all(
    chunk_results: dict[str, list[Chunk]],
    embedding_client: EmbeddingClient | MockEmbeddingClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    calculate_cs: bool = True,
    verbose: bool = False
) -> dict[str, ChunkingMetrics]:
    """Evaluate chunking quality for all parsers using semantic distance.

    Args:
        chunk_results: Dictionary mapping parser name to chunks
        embedding_client: Embedding client for semantic distance calculation
        threshold_k: CS threshold (similarity >= threshold keeps edge)
        graph_type: CS graph type ("complete" or "incomplete")
        calculate_cs: Whether to calculate CS (can be slow)
        verbose: Print verbose output

    Returns:
        Dictionary mapping parser name to ChunkingMetrics
    """
    if embedding_client is None:
        print("Note: Using MockEmbeddingClient (no model loaded)")
        embedding_client = MockEmbeddingClient()

    evaluation = {}

    for parser_name, chunks in chunk_results.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {parser_name} ({len(chunks)} chunks)")
        print(f"{'='*60}")

        if len(chunks) < 2:
            print("  Skip: Need at least 2 chunks for evaluation")
            continue

        metrics = evaluate_chunking(
            chunks=chunks,
            embedding_client=embedding_client,
            threshold_k=threshold_k,
            graph_type=graph_type,
            calculate_cs_flag=calculate_cs,
            verbose=verbose
        )

        evaluation[parser_name] = metrics

        # Print results
        if metrics.bc_score:
            bc = metrics.bc_score
            print(f"\n  BC (Boundary Clarity): {bc.score:.4f}")
            print("     - Higher is better (chunks are independent)")
            print(f"     - Range: [{bc.min_score:.4f}, {bc.max_score:.4f}]")
            print(f"     - Pairs evaluated: {bc.pair_count}")

        if metrics.cs_score:
            cs = metrics.cs_score
            print(f"\n  CS (Chunk Stickiness): {cs.score:.4f}")
            print("     - Lower is better (less inter-chunk dependency)")
            print(f"     - Graph: {cs.graph_type} ({cs.node_count} nodes, {cs.edge_count} edges)")
            print(f"     - Threshold K: {cs.threshold_k}")

    return evaluation


def print_comparison_table(evaluation: dict[str, ChunkingMetrics]):
    """Print comparison table of all parsers."""
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)

    # Header
    print(f"\n| {'Parser':<12} | {'BC (↑)':<10} | {'CS (↓)':<10} | {'Chunks':<8} |")
    print(f"|{'-'*14}|{'-'*12}|{'-'*12}|{'-'*10}|")

    # Rows
    for parser_name, metrics in evaluation.items():
        bc_str = f"{metrics.bc_score.score:.4f}" if metrics.bc_score else "N/A"
        cs_str = f"{metrics.cs_score.score:.4f}" if metrics.cs_score else "N/A"
        chunk_count = metrics.bc_score.pair_count + 1 if metrics.bc_score else 0

        print(f"| {parser_name:<12} | {bc_str:<10} | {cs_str:<10} | {chunk_count:<8} |")

    # Reference values from MoC paper
    print("\n  Reference (MoC Paper - Qwen2.5-7B):")
    print("  | Method      | BC (↑)  | CS_i (↓) |")
    print("  |-------------|---------|----------|")
    print("  | Fixed       | 0.8049  | 1.898    |")
    print("  | Llama_index | 0.8455  | 1.483    |")
    print("  | Semantic    | 0.8140  | 1.650    |")
    print("  | LLM         | 0.8641  | 1.438    |")


# =============================================================================
# Output
# =============================================================================

def save_chunking_json(
    chunk_results: dict[str, list[Chunk]],
    evaluation: dict[str, ChunkingMetrics],
    output_dir: Path,
    config: dict
):
    """Save chunking evaluation results to chunking.json

    Args:
        chunk_results: Chunked results per parser
        evaluation: Evaluation metrics per parser
        output_dir: Output directory (same as parsing results)
        config: Configuration used for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build chunking.json structure
    chunking_data = {
        "timestamp": timestamp,
        "config": config,
        "results": {}
    }

    for parser_name, metrics in evaluation.items():
        chunks = chunk_results.get(parser_name, [])

        parser_result = {
            "chunk_count": len(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }

        if metrics.bc_score:
            parser_result["bc"] = {
                "score": metrics.bc_score.score,
                "min": metrics.bc_score.min_score,
                "max": metrics.bc_score.max_score,
                "std": metrics.bc_score.std_dev,
                "pair_count": metrics.bc_score.pair_count,
            }

        if metrics.cs_score:
            parser_result["cs"] = {
                "score": metrics.cs_score.score,
                "graph_type": metrics.cs_score.graph_type,
                "node_count": metrics.cs_score.node_count,
                "edge_count": metrics.cs_score.edge_count,
                "threshold_k": metrics.cs_score.threshold_k,
            }

        chunking_data["results"][parser_name] = parser_result

    # Save to chunking.json
    chunking_file = output_dir / "chunking.json"
    chunking_file.write_text(
        json.dumps(chunking_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  ✓ Saved: {chunking_file}")

    return chunking_file


# =============================================================================
# Batch Processing
# =============================================================================

def scan_results_folders(results_dir: Path = Path("results")) -> list[dict]:
    """Scan results/ folder for all test_* folders with parsing results.

    Returns:
        List of dicts with keys: test_id, folder_path, parsers
    """
    test_folders = []

    if not results_dir.exists():
        print(f"❌ 결과 폴더를 찾을 수 없습니다: {results_dir}")
        return []

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("test_"):
            continue

        test_id = folder.name

        # Check for parsing results
        evaluation_file = folder / "evaluation.json"
        if not evaluation_file.exists():
            print(f"⚠️ {test_id}: evaluation.json 없음 (파싱 먼저 실행 필요)")
            continue

        # Find parsed output files (both old and new naming conventions)
        parsers = []
        parser_patterns = [
            # New naming convention
            ("image_advanced", "Image-Advanced"),
            ("image_baseline", "Image-Baseline"),
            ("text_advanced", "Text-Advanced"),
            ("text_baseline", "Text-Baseline"),
            # Old naming convention
            ("vlm", "VLM"),
            ("ocr-text", "OCR-Text"),
            ("ocr-image", "OCR-Image"),
            ("twostage-text", "TwoStage-Text"),
            ("twostage-image", "TwoStage-Image"),
        ]
        for pattern, display_name in parser_patterns:
            for ext in [".txt", ".md"]:
                output_file = folder / f"{pattern}_output{ext}"
                if output_file.exists():
                    content = output_file.read_text(encoding="utf-8")
                    if len(content.strip()) > 0:
                        parsers.append({
                            "name": display_name,
                            "file": output_file,
                            "content_length": len(content)
                        })
                    break

        if parsers:
            test_folders.append({
                "test_id": test_id,
                "folder_path": folder,
                "parsers": parsers
            })
        else:
            print(f"⚠️ {test_id}: 파싱 결과 파일 없음")

    return test_folders


def run_single_chunking_test(
    test_folder: Path,
    breakpoint_type: str = "percentile",
    breakpoint_threshold: float = 95.0,
    min_chunk_size: Optional[int] = None,
    embedding_api_url: str = "http://localhost:8001/embeddings",
    embedding_model: str = "BAAI/bge-m3",
    embedding_client: EmbeddingClient | MockEmbeddingClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    calculate_cs: bool = True,
    verbose: bool = False
) -> dict:
    """Run chunking evaluation on a single test folder.

    Args:
        test_folder: Path to results/test_X folder
        Other args: Chunking configuration

    Returns:
        dict with chunk_results and evaluation
    """
    # Load parsed files
    chunk_results = {}

    # Support both old and new naming conventions
    parser_patterns = [
        # New naming convention
        ("image_advanced", "Image-Advanced"),
        ("image_baseline", "Image-Baseline"),
        ("text_advanced", "Text-Advanced"),
        ("text_baseline", "Text-Baseline"),
        # Old naming convention
        ("vlm", "VLM"),
        ("ocr-text", "OCR-Text"),
        ("ocr-image", "OCR-Image"),
        ("twostage-text", "TwoStage-Text"),
        ("twostage-image", "TwoStage-Image"),
    ]

    for pattern, parser_name in parser_patterns:
        for ext in [".txt", ".md"]:
            output_file = test_folder / f"{pattern}_output{ext}"
            if output_file.exists():
                content = output_file.read_text(encoding="utf-8")
                if len(content.strip()) > 100:  # Minimum content length
                    chunks = chunk_text(
                        content,
                        breakpoint_type=breakpoint_type,
                        breakpoint_threshold=breakpoint_threshold,
                        min_chunk_size=min_chunk_size,
                        embedding_api_url=embedding_api_url,
                        embedding_model=embedding_model,
                        document_id=pattern
                    )
                    chunk_results[parser_name] = chunks
                    print(f"  {parser_name}: {len(chunks)} chunks")
                break

    if not chunk_results:
        print("  ❌ No valid content to chunk")
        return {"chunk_results": {}, "evaluation": {}}

    # Evaluate
    evaluation = evaluate_all(
        chunk_results,
        embedding_client=embedding_client,
        threshold_k=threshold_k,
        graph_type=graph_type,
        calculate_cs=calculate_cs,
        verbose=verbose
    )

    return {"chunk_results": chunk_results, "evaluation": evaluation}


def run_all_chunking_tests(
    results_dir: Path = Path("results"),
    breakpoint_type: str = "percentile",
    breakpoint_threshold: float = 95.0,
    min_chunk_size: Optional[int] = None,
    embedding_model: str = "BAAI/bge-m3",
    embedding_api_url: str = "http://localhost:8001/embeddings",
    use_mock: bool = False,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    skip_cs: bool = False,
    verbose: bool = False
) -> dict:
    """Run chunking evaluation on all test folders in results/.

    Returns:
        dict mapping test_id to results
    """
    test_folders = scan_results_folders(results_dir)

    if not test_folders:
        print("❌ 청킹 테스트할 데이터가 없습니다.")
        print("   먼저 파싱 테스트를 실행하세요: python -m src.eval_parsers --all")
        return {}

    print("=" * 60)
    print("🔬 Chunking Evaluation - Batch Test (Semantic Distance)")
    print("=" * 60)
    print(f"📁 결과 폴더: {results_dir}")
    print(f"📊 테스트 수: {len(test_folders)}")
    print(f"📐 Breakpoint Type: {breakpoint_type}")
    print(f"🎯 Breakpoint Threshold: {breakpoint_threshold}")
    if min_chunk_size:
        print(f"📏 Min Chunk Size: {min_chunk_size}")
    print()

    for info in test_folders:
        parser_names = [p["name"] for p in info["parsers"]]
        print(f"  - {info['test_id']}: {', '.join(parser_names)}")

    # Create embedding client
    embedding_client = create_embedding_client(
        model=embedding_model,
        use_mock=use_mock,
        api_url=embedding_api_url if embedding_api_url else None
    )
    if use_mock:
        print("\n⚠️ MockEmbeddingClient 사용 (단어 기반 근사값)")
    elif embedding_api_url:
        print(f"\n🔤 Embedding API: {embedding_api_url}")
        print(f"   Model: {embedding_model}")
    else:
        print(f"\n🔤 Local Embedding model: {embedding_model}")

    all_results = {}

    config = {
        "breakpoint_type": breakpoint_type,
        "breakpoint_threshold": breakpoint_threshold,
        "min_chunk_size": min_chunk_size,
        "graph_type": graph_type,
        "threshold_k": threshold_k,
        "embedding_model": embedding_model,
        "embedding_api_url": embedding_api_url,
        "use_mock": use_mock,
    }

    for i, info in enumerate(test_folders, 1):
        test_id = info["test_id"]
        folder = info["folder_path"]

        print()
        print("#" * 60)
        print(f"# [{i}/{len(test_folders)}] {test_id}")
        print("#" * 60)

        result = run_single_chunking_test(
            test_folder=folder,
            breakpoint_type=breakpoint_type,
            breakpoint_threshold=breakpoint_threshold,
            min_chunk_size=min_chunk_size,
            embedding_api_url=embedding_api_url,
            embedding_model=embedding_model,
            embedding_client=embedding_client,
            threshold_k=threshold_k,
            graph_type=graph_type,
            calculate_cs=not skip_cs,
            verbose=verbose
        )

        # Save chunking.json in the same folder
        if result["evaluation"]:
            save_chunking_json(
                result["chunk_results"],
                result["evaluation"],
                folder,
                config
            )

        all_results[test_id] = result

    # Summary
    print()
    print("=" * 60)
    print("📊 전체 Chunking 평가 요약")
    print("=" * 60)

    print(f"\n| Test ID | Parser | BC (↑) | CS (↓) | Chunks |")
    print(f"|---------|--------|--------|--------|--------|")

    for test_id, result in all_results.items():
        evaluation = result.get("evaluation", {})
        for parser_name, metrics in evaluation.items():
            bc_str = f"{metrics.bc_score.score:.4f}" if metrics.bc_score else "N/A"
            cs_str = f"{metrics.cs_score.score:.4f}" if metrics.cs_score else "N/A"
            chunk_count = metrics.bc_score.pair_count + 1 if metrics.bc_score else 0
            print(f"| {test_id:<8} | {parser_name:<6} | {bc_str:<6} | {cs_str:<6} | {chunk_count:<6} |")

    return all_results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chunking Evaluation CLI - Semantic Distance-based BC/CS metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics (Semantic Distance based):
  BC (Boundary Clarity):  Higher is better - chunks are semantically independent
  CS (Chunk Stickiness):  Lower is better - less semantic similarity between chunks

Breakpoint Types:
  percentile:         Use percentile of distances (default, threshold ~95)
  standard_deviation: Use standard deviations above mean (threshold ~3)
  interquartile:      Use interquartile range (threshold ~1.5)
  gradient:           Use gradient of distances (threshold ~95)

Examples:
  # 전체 테스트 (results/ 폴더의 모든 파싱 결과)
  python -m src.eval_chunking --all

  # 특정 테스트 폴더
  python -m src.eval_chunking --parsed-dir results/test_1/

  # 다른 breakpoint 설정
  python -m src.eval_chunking --all --breakpoint-type standard_deviation --breakpoint-threshold 3.0
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="results/ 폴더의 모든 test_* 결과 청킹 테스트"
    )
    input_group.add_argument(
        "--parsed-dir",
        type=Path,
        help="Directory with pre-parsed output files"
    )
    input_group.add_argument(
        "--parsed-files",
        type=Path,
        nargs="+",
        help="Specific parsed output files"
    )

    # Directory options
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="결과 폴더 (기본: results/)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results"
    )

    # Semantic chunking options
    parser.add_argument(
        "--breakpoint-type",
        choices=["percentile", "standard_deviation", "interquartile", "gradient"],
        default="percentile",
        help="Breakpoint detection type (default: percentile)"
    )
    parser.add_argument(
        "--breakpoint-threshold",
        type=float,
        default=95.0,
        help="Breakpoint threshold amount (default: 95.0 for percentile)"
    )
    parser.add_argument(
        "--min-chunk-size",
        type=int,
        default=None,
        help="Minimum chunk size in characters (optional)"
    )

    # Embedding model options
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="Model name for embeddings (default: BAAI/bge-m3)"
    )
    parser.add_argument(
        "--embedding-api-url",
        default="http://localhost:8001/embeddings",
        help="Embedding API URL (Infinity/OpenAI compatible). Set to empty string to use local model."
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock embedding client (word-based heuristic, for testing)"
    )

    # CS options
    parser.add_argument(
        "--graph-type",
        choices=["complete", "incomplete"],
        default="incomplete",
        help="CS graph type (default: incomplete)"
    )
    parser.add_argument(
        "--threshold-k",
        type=float,
        default=0.8,
        help="CS edge filtering threshold (default: 0.8)"
    )
    parser.add_argument(
        "--skip-cs",
        action="store_true",
        help="Skip CS calculation (faster, BC only)"
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # --all 모드: 전체 테스트
    if args.all:
        run_all_chunking_tests(
            results_dir=args.results_dir,
            breakpoint_type=args.breakpoint_type,
            breakpoint_threshold=args.breakpoint_threshold,
            min_chunk_size=args.min_chunk_size,
            embedding_model=args.embedding_model,
            embedding_api_url=args.embedding_api_url,
            use_mock=args.use_mock,
            threshold_k=args.threshold_k,
            graph_type=args.graph_type,
            skip_cs=args.skip_cs,
            verbose=args.verbose
        )
        print("\nDone!")
        return

    # Print header
    print("=" * 60)
    print("Chunking Evaluation CLI")
    print("MoC-based Quality Metrics (BC + CS)")
    print("=" * 60)
    print()

    # Get chunk results
    chunk_results = {}

    if args.parsed_dir:
        # Load from pre-parsed directory
        if not args.parsed_dir.exists():
            print(f"Error: Parsed directory not found: {args.parsed_dir}")
            sys.exit(1)

        print(f"Loading parsed files from: {args.parsed_dir}")
        chunk_results = load_parsed_files(
            args.parsed_dir,
            breakpoint_type=args.breakpoint_type,
            breakpoint_threshold=args.breakpoint_threshold,
            min_chunk_size=args.min_chunk_size,
            embedding_api_url=args.embedding_api_url,
            embedding_model=args.embedding_model,
        )

    elif args.parsed_files:
        # Load from specific files
        for file_path in args.parsed_files:
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            parser_name = file_path.stem.replace("_output", "").upper()
            content = file_path.read_text(encoding="utf-8")

            chunks = chunk_text(
                content,
                breakpoint_type=args.breakpoint_type,
                breakpoint_threshold=args.breakpoint_threshold,
                min_chunk_size=args.min_chunk_size,
                embedding_api_url=args.embedding_api_url,
                embedding_model=args.embedding_model,
                document_id=parser_name.lower()
            )
            chunk_results[parser_name] = chunks
            print(f"Loaded {parser_name}: {len(chunks)} chunks")

    if not chunk_results:
        print("Error: No chunks to evaluate")
        sys.exit(1)

    # Create embedding client
    embedding_client = create_embedding_client(
        model=args.embedding_model,
        use_mock=args.use_mock,
        api_url=args.embedding_api_url if args.embedding_api_url else None
    )

    if args.use_mock:
        print("\nNote: Using MockEmbeddingClient (word-based heuristic)")
    elif args.embedding_api_url:
        print(f"\n🔤 Embedding API: {args.embedding_api_url}")
        print(f"   Model: {args.embedding_model}")
    else:
        print(f"\n🔤 Local Embedding model: {args.embedding_model}")

    # Evaluate
    evaluation = evaluate_all(
        chunk_results,
        embedding_client=embedding_client,
        threshold_k=args.threshold_k,
        graph_type=args.graph_type,
        calculate_cs=not args.skip_cs,
        verbose=args.verbose
    )

    # Print comparison table
    print_comparison_table(evaluation)

    # Save results
    output_dir = args.output_dir or args.parsed_dir
    if output_dir:
        config = {
            "breakpoint_type": args.breakpoint_type,
            "breakpoint_threshold": args.breakpoint_threshold,
            "min_chunk_size": args.min_chunk_size,
            "graph_type": args.graph_type,
            "threshold_k": args.threshold_k,
            "embedding_model": args.embedding_model,
            "embedding_api_url": args.embedding_api_url,
            "use_mock": args.use_mock,
        }

        print(f"\nSaving results to: {output_dir}")
        save_chunking_json(chunk_results, evaluation, output_dir, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
