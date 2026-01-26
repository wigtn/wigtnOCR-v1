#!/usr/bin/env python3
"""
Chunking Evaluation CLI

MoC-based chunking quality evaluation for VLM document parsing.
Evaluates chunking quality using BC (Boundary Clarity) and CS (Chunk Stickiness)
metrics that don't require Ground Truth labels.

Reference: MoC Paper (arXiv:2503.09600v2)

Usage:
    # Basic evaluation with PDF input
    python -m src.test_chunking --input data/test.pdf

    # With specific chunking strategy
    python -m src.test_chunking --input data/test.pdf --strategy recursive_character

    # With existing parsed files
    python -m src.test_chunking --parsed-dir results/parsing/test_1/

    # Skip VLM parser (use only OCR)
    python -m src.test_chunking --input data/test.pdf --skip-vlm

    # Full evaluation with CS graph
    python -m src.test_chunking --input data/test.pdf --graph-type complete --threshold-k 0.7
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
#   1. python -m src.test_chunking (프로젝트 루트에서)
#   2. python test_chunking.py (src/ 디렉토리에서)

try:
    # 방법 1: src.xxx (프로젝트 루트에서 실행)
    from src.chunking.chunker import (
        ChunkerConfig,
        ChunkingStrategy,
        create_chunker,
        Chunk,
    )
    from src.chunking.metrics import (
        LLMClient,
        MockLLMClient,
        evaluate_chunking,
        ChunkingMetrics,
    )
    from src.test_parsers import (
        FileFormat,
        detect_file_format,
        test_vlm_parser,
        test_ocr_text_parser,
        test_ocr_image_parser,
        convert_hwp_to_images,
    )
except ImportError:
    # 방법 2: xxx (src/ 디렉토리에서 실행 또는 PYTHONPATH 설정)
    from chunking.chunker import (
        ChunkerConfig,
        ChunkingStrategy,
        create_chunker,
        Chunk,
    )
    from chunking.metrics import (
        LLMClient,
        MockLLMClient,
        evaluate_chunking,
        ChunkingMetrics,
    )
    from test_parsers import (
        FileFormat,
        detect_file_format,
        test_vlm_parser,
        test_ocr_text_parser,
        test_ocr_image_parser,
        convert_hwp_to_images,
    )


# =============================================================================
# Chunking Pipeline
# =============================================================================

def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    document_id: str = "doc"
) -> list[Chunk]:
    """Chunk text using the specified strategy.

    Args:
        text: Input text to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        document_id: Document identifier for chunk IDs

    Returns:
        List of Chunk objects
    """
    config = ChunkerConfig(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunker = create_chunker(config)
    return chunker.chunk(text, document_id)


def parse_and_chunk(
    input_path: Path,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE_CHARACTER,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    skip_vlm: bool = False,
    skip_docling: bool = False,
    verbose: bool = False
) -> dict[str, list[Chunk]]:
    """Parse document and chunk the results.

    Args:
        input_path: Path to input file (PDF, image, HWP)
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        skip_vlm: Skip VLM parser
        skip_docling: Skip Docling parser
        verbose: Print verbose output

    Returns:
        Dictionary mapping parser name to list of chunks
    """
    file_format = detect_file_format(input_path)
    input_bytes = input_path.read_bytes()

    print(f"Input: {input_path}")
    print(f"Format: {file_format.value.upper()}")
    print(f"Size: {len(input_bytes) / 1024:.1f} KB")
    print()

    results = {}
    parse_results = {}

    # HWP preprocessing
    hwp_images = None
    if file_format in [FileFormat.HWP, FileFormat.HWPX]:
        print("Converting HWP to images...")
        hwp_images = convert_hwp_to_images(input_path)
        if not hwp_images:
            print("Error: HWP conversion failed")
            return {}

    # Parse with each parser
    if file_format == FileFormat.PDF:
        # PDF: All parsers
        if not skip_vlm:
            try:
                parse_results["VLM"] = test_vlm_parser(input_bytes, verbose, FileFormat.PDF)
            except Exception as e:
                print(f"VLM Parser error: {e}")

        try:
            parse_results["OCR-Text"] = test_ocr_text_parser(input_bytes, verbose)
        except Exception as e:
            print(f"OCR-Text Parser error: {e}")

        if not skip_docling:
            try:
                parse_results["OCR-Image"] = test_ocr_image_parser(input_bytes, verbose)
            except Exception as e:
                print(f"OCR-Image Parser error: {e}")

    elif file_format == FileFormat.IMAGE:
        # Image: VLM only
        if skip_vlm:
            print("Error: Image input requires VLM parser")
            return {}

        try:
            parse_results["VLM"] = test_vlm_parser(input_bytes, verbose, FileFormat.IMAGE)
        except Exception as e:
            print(f"VLM Parser error: {e}")

    elif file_format in [FileFormat.HWP, FileFormat.HWPX]:
        # HWP: VLM only (via converted images)
        if skip_vlm:
            print("Error: HWP input requires VLM parser")
            return {}

        try:
            parse_results["VLM"] = test_vlm_parser(
                input_bytes, verbose, file_format, pre_converted_images=hwp_images
            )
        except Exception as e:
            print(f"VLM Parser error: {e}")

    # Chunk each parser's output
    print()
    print("=" * 60)
    print("Chunking Results")
    print("=" * 60)

    for parser_name, parse_result in parse_results.items():
        if not parse_result.get("success"):
            print(f"{parser_name}: SKIP (parsing failed)")
            continue

        content = parse_result.get("content", "")
        if not content:
            print(f"{parser_name}: SKIP (no content)")
            continue

        # Chunk the content
        chunks = chunk_text(
            content,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            document_id=f"{parser_name.lower()}"
        )

        results[parser_name] = chunks
        print(f"{parser_name}: {len(chunks)} chunks (avg {sum(c.length for c in chunks) // max(len(chunks), 1)} chars)")

    return results


def load_parsed_files(parsed_dir: Path) -> dict[str, list[Chunk]]:
    """Load pre-parsed files from a directory and chunk them.

    Expected file naming:
    - vlm_output.md or vlm_output.txt
    - ocr-text_output.md or ocr-text_output.txt
    - ocr-image_output.md or ocr-image_output.txt

    Args:
        parsed_dir: Directory containing parsed output files

    Returns:
        Dictionary mapping parser name to list of chunks
    """
    results = {}

    # Map filename patterns to parser names
    patterns = {
        "vlm": "VLM",
        "ocr-text": "OCR-Text",
        "ocr-image": "OCR-Image",
    }

    for pattern, parser_name in patterns.items():
        # Try different extensions
        for ext in [".md", ".txt"]:
            file_path = parsed_dir / f"{pattern}_output{ext}"
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                chunks = chunk_text(content, document_id=pattern)
                results[parser_name] = chunks
                print(f"Loaded {parser_name}: {len(chunks)} chunks from {file_path.name}")
                break

    return results


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_all(
    chunk_results: dict[str, list[Chunk]],
    llm_client: LLMClient | MockLLMClient | None = None,
    threshold_k: float = 0.8,
    graph_type: str = "incomplete",
    calculate_cs: bool = True,
    verbose: bool = False
) -> dict[str, ChunkingMetrics]:
    """Evaluate chunking quality for all parsers.

    Args:
        chunk_results: Dictionary mapping parser name to chunks
        llm_client: LLM client for perplexity calculation
        threshold_k: CS threshold
        graph_type: CS graph type ("complete" or "incomplete")
        calculate_cs: Whether to calculate CS (can be slow)
        verbose: Print verbose output

    Returns:
        Dictionary mapping parser name to ChunkingMetrics
    """
    if llm_client is None:
        print("Note: Using MockLLMClient (no API configured)")
        llm_client = MockLLMClient()

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
            llm_client=llm_client,
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

def save_results(
    chunk_results: dict[str, list[Chunk]],
    evaluation: dict[str, ChunkingMetrics],
    output_dir: Path,
    config: dict
):
    """Save evaluation results to files.

    Args:
        chunk_results: Chunked results per parser
        evaluation: Evaluation metrics per parser
        output_dir: Output directory
        config: Configuration used for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Save chunks as JSON for each parser
    for parser_name, chunks in chunk_results.items():
        safe_name = parser_name.lower().replace("-", "_")
        chunks_file = output_dir / f"{safe_name}_chunks.json"

        chunks_data = [c.to_dict() for c in chunks]
        chunks_file.write_text(
            json.dumps(chunks_data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  Saved: {chunks_file.name}")

    # 2. Save evaluation results
    eval_data = {
        "timestamp": timestamp,
        "config": config,
        "results": {}
    }

    for parser_name, metrics in evaluation.items():
        eval_data["results"][parser_name] = metrics.to_dict()

    eval_file = output_dir / "evaluation.json"
    eval_file.write_text(
        json.dumps(eval_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  Saved: {eval_file.name}")

    # 3. Save summary README
    readme_lines = [
        "# Chunking Evaluation Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- Strategy: {config.get('strategy', 'recursive_character')}",
        f"- Chunk Size: {config.get('chunk_size', 500)}",
        f"- Chunk Overlap: {config.get('chunk_overlap', 50)}",
        f"- Graph Type: {config.get('graph_type', 'incomplete')}",
        f"- Threshold K: {config.get('threshold_k', 0.8)}",
        "",
        "## Results",
        "",
        "| Parser | BC (↑) | CS (↓) | Chunks |",
        "|--------|--------|--------|--------|",
    ]

    for parser_name, metrics in evaluation.items():
        bc_str = f"{metrics.bc_score.score:.4f}" if metrics.bc_score else "N/A"
        cs_str = f"{metrics.cs_score.score:.4f}" if metrics.cs_score else "N/A"
        chunk_count = metrics.bc_score.pair_count + 1 if metrics.bc_score else 0
        readme_lines.append(f"| {parser_name} | {bc_str} | {cs_str} | {chunk_count} |")

    readme_lines.extend([
        "",
        "## Metrics Explanation",
        "",
        "- **BC (Boundary Clarity)**: Higher is better. Measures chunk independence.",
        "- **CS (Chunk Stickiness)**: Lower is better. Measures inter-chunk dependency.",
        "",
        "Reference: MoC Paper (arXiv:2503.09600v2)",
    ])

    readme_file = output_dir / "README.md"
    readme_file.write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"  Saved: {readme_file.name}")


# =============================================================================
# CLI
# =============================================================================

def create_llm_client(
    llm_model: str,
    llm_api_url: Optional[str] = None,
    use_mock: bool = False
) -> LLMClient | MockLLMClient:
    """Create LLM client based on configuration.

    Args:
        llm_model: Model identifier
        llm_api_url: API URL (optional, uses default if not provided)
        use_mock: Use mock client instead of real API

    Returns:
        LLM client instance
    """
    if use_mock:
        return MockLLMClient()

    # Default API URLs based on model
    default_urls = {
        "qwen2.5-7b": "http://localhost:8000/v1/completions",
        "qwen2.5-14b": "http://localhost:8000/v1/completions",
        "gpt-3.5-turbo": "https://api.openai.com/v1/completions",
        "gpt-4": "https://api.openai.com/v1/completions",
    }

    api_url = llm_api_url or default_urls.get(llm_model.lower(), "http://localhost:8000/v1/completions")

    return LLMClient(
        api_url=api_url,
        model=llm_model,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Chunking Evaluation CLI - MoC-based quality metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  BC (Boundary Clarity):  Higher is better - chunks are independent
  CS (Chunk Stickiness):  Lower is better - less inter-chunk dependency

Examples:
  python -m src.test_chunking --input data/test.pdf
  python -m src.test_chunking --parsed-dir results/parsing/test_1/
  python -m src.test_chunking --input data/test.pdf --use-mock
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        help="Input file (PDF, image, HWP/HWPX)"
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

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory for results (default: results/chunks/<timestamp>)"
    )

    # Chunking options
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive_character", "semantic", "hierarchical"],
        default="recursive_character",
        help="Chunking strategy (default: recursive_character)"
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Compare multiple strategies (comma-separated: fixed,recursive_character,semantic)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Overlap between chunks (default: 50)"
    )

    # Parser options
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM parser"
    )
    parser.add_argument(
        "--skip-docling",
        action="store_true",
        help="Skip Docling (OCR-Image) parser"
    )

    # LLM options
    parser.add_argument(
        "--llm-model",
        default="qwen2.5-7b",
        help="LLM model for perplexity calculation (default: qwen2.5-7b)"
    )
    parser.add_argument(
        "--llm-api-url",
        default=None,
        help="LLM API URL (default: auto-detect based on model)"
    )
    parser.add_argument(
        "--use-mock",
        action="store_true",
        help="Use mock LLM client (no API required, for testing)"
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

    # Print header
    print("=" * 60)
    print("Chunking Evaluation CLI")
    print("MoC-based Quality Metrics (BC + CS)")
    print("=" * 60)
    print()

    # Get chunk results
    chunk_results = {}

    if args.input:
        # Parse and chunk from input file
        if not args.input.exists():
            print(f"Error: Input file not found: {args.input}")
            sys.exit(1)

        strategy = ChunkingStrategy(args.strategy)
        chunk_results = parse_and_chunk(
            args.input,
            strategy=strategy,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            skip_vlm=args.skip_vlm,
            skip_docling=args.skip_docling,
            verbose=args.verbose
        )

    elif args.parsed_dir:
        # Load from pre-parsed directory
        if not args.parsed_dir.exists():
            print(f"Error: Parsed directory not found: {args.parsed_dir}")
            sys.exit(1)

        print(f"Loading parsed files from: {args.parsed_dir}")
        chunk_results = load_parsed_files(args.parsed_dir)

    elif args.parsed_files:
        # Load from specific files
        for file_path in args.parsed_files:
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue

            parser_name = file_path.stem.replace("_output", "").upper()
            content = file_path.read_text(encoding="utf-8")

            # 다중 전략 비교 모드
            if args.strategies:
                strategies = [s.strip() for s in args.strategies.split(",")]
                for strat_name in strategies:
                    try:
                        strat = ChunkingStrategy(strat_name)
                        chunks = chunk_text(
                            content,
                            strategy=strat,
                            chunk_size=args.chunk_size,
                            chunk_overlap=args.chunk_overlap,
                            document_id=f"{parser_name.lower()}_{strat_name}"
                        )
                        result_name = f"{parser_name}_{strat_name}"
                        chunk_results[result_name] = chunks
                        print(f"Loaded {result_name}: {len(chunks)} chunks")
                    except ValueError:
                        print(f"Warning: Invalid strategy '{strat_name}'")
            else:
                chunks = chunk_text(
                    content,
                    strategy=ChunkingStrategy(args.strategy),
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    document_id=parser_name.lower()
                )
                chunk_results[parser_name] = chunks
                print(f"Loaded {parser_name}: {len(chunks)} chunks")

    if not chunk_results:
        print("Error: No chunks to evaluate")
        sys.exit(1)

    # Create LLM client
    llm_client = create_llm_client(
        args.llm_model,
        args.llm_api_url,
        args.use_mock
    )

    if args.use_mock:
        print("\nNote: Using MockLLMClient (results are approximate)")

    # Evaluate
    evaluation = evaluate_all(
        chunk_results,
        llm_client=llm_client,
        threshold_k=args.threshold_k,
        graph_type=args.graph_type,
        calculate_cs=not args.skip_cs,
        verbose=args.verbose
    )

    # Print comparison table
    print_comparison_table(evaluation)

    # Save results
    if args.output_dir or args.input:
        output_dir = args.output_dir
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("results/chunks") / timestamp

        config = {
            "strategy": args.strategy,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "graph_type": args.graph_type,
            "threshold_k": args.threshold_k,
            "llm_model": args.llm_model,
            "use_mock": args.use_mock,
        }

        print(f"\nSaving results to: {output_dir}")
        save_results(chunk_results, evaluation, output_dir, config)

    print("\nDone!")


if __name__ == "__main__":
    main()
