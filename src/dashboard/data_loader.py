"""
Data loading utilities for VLM Document Parsing Dashboard.

Supports loading test results from JSON files with caching.

Usage:
    from dashboard.data_loader import load_results, load_results_cached

    # With Streamlit caching
    data = load_results_cached()

    # Without caching (for CLI usage)
    data = load_results()
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RESULTS_PATH = Path(__file__).parent.parent.parent / "results" / "parsing_results.json"
CACHE_TTL = 3600  # 1 hour


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParsedData:
    """Parsed and structured test results."""
    version: str
    created_at: str
    parsers: List[str]
    parsing_results: List[Dict[str, Any]]
    chunking_results: List[Dict[str, Any]]
    raw_data: Dict[str, Any]

    @property
    def total_tests(self) -> int:
        return len(self.parsing_results)

    @property
    def total_chunks(self) -> int:
        return sum(len(r.get("chunks", [])) for r in self.chunking_results)


# =============================================================================
# Core Loading Functions
# =============================================================================

def load_results(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load test results from JSON file.

    Args:
        results_path: Path to JSON file. Uses default if not specified.

    Returns:
        Dict with keys: version, created_at, test_config, parsing_results, chunking_results
        Or dict with "error" key if file not found.
    """
    path = results_path or DEFAULT_RESULTS_PATH

    if not path.exists():
        return {
            "error": f"결과 파일이 없습니다: {path}\nCLI에서 테스트를 먼저 실행하세요.",
            "path": str(path),
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON 파싱 오류: {e}",
            "path": str(path),
        }
    except Exception as e:
        return {
            "error": f"파일 로드 실패: {e}",
            "path": str(path),
        }


def load_results_cached(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load test results with Streamlit caching.

    Uses @st.cache_data with 1-hour TTL for performance.
    Falls back to non-cached version if Streamlit not available.
    """
    if not HAS_STREAMLIT:
        return load_results(results_path)

    # Define cached function inside to avoid decorator issues
    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def _cached_load(path_str: str) -> Dict[str, Any]:
        return load_results(Path(path_str))

    path = results_path or DEFAULT_RESULTS_PATH
    return _cached_load(str(path))


# =============================================================================
# Data Transformation Functions
# =============================================================================

def get_parsing_data(data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Transform parsing results to dashboard-friendly format.

    Returns:
        Dict keyed by test_id with structure:
        {
            "test_1": {
                "id": "Test 1",
                "name": "...",
                "doc_type": "...",
                "parsers": {
                    "VLM (Qwen3-VL)": {"wer": 0.1, "cer": 0.08, ...},
                    ...
                }
            }
        }
    """
    if "error" in data:
        return {}

    result = {}
    for item in data.get("parsing_results", []):
        test_id = item.get("test_id", "unknown")
        result[test_id] = {
            "id": test_id.replace("_", " ").title(),
            "name": item.get("name", test_id),
            "doc_type": item.get("doc_type", "unknown"),
            "source_path": item.get("source_path", ""),
            "ground_truth_path": item.get("ground_truth_path", ""),
            "parsers": item.get("results", {}),
        }
    return result


def get_chunking_data(data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Transform chunking results to dashboard-friendly format.

    Supports both legacy (List) and new (Dict, per-parser) formats.

    Legacy format (v1.0):
        chunking_results: [{strategy: "Semantic", ...}, ...]

    New format (v1.1):
        chunking_results: {
            "VLM (Qwen3-VL)": {parser: ..., strategies: [...]},
            ...
        }

    Returns:
        Dict keyed by parser with structure (new format):
        {
            "VLM (Qwen3-VL)": {
                "parser": "VLM (Qwen3-VL)",
                "strategies": [
                    {"strategy": "Semantic", "mean_bc": 0.88, ...},
                    ...
                ]
            }
        }

        Or legacy format (backward compatible):
        {
            "_legacy": {
                "parser": "_legacy",
                "strategies": [
                    {"strategy": "Semantic", ...},
                    ...
                ]
            }
        }
    """
    if "error" in data:
        return {}

    chunking_results = data.get("chunking_results", {})

    # Legacy format detection (List)
    if isinstance(chunking_results, list):
        strategies = []
        for item in chunking_results:
            strategies.append({
                "strategy": item.get("strategy", "unknown"),
                "params": item.get("params", {}),
                "chunks": item.get("chunks", []),
                "mean_bc": item.get("mean_bc", 0.0),
                "mean_cs": item.get("mean_cs", 0.0),
                "std_bc": item.get("std_bc"),
                "std_cs": item.get("std_cs"),
                "bc_by_sentence": item.get("bc_by_sentence", []),
            })
        return {"_legacy": {"parser": "_legacy", "strategies": strategies}}

    # New format (Dict, per-parser)
    return chunking_results


def get_chunking_data_for_parser(data: Dict[str, Any], parser: str) -> List[Dict]:
    """
    Get chunking strategies for a specific parser.

    Args:
        data: Raw JSON data
        parser: Parser name (e.g., "VLM (Qwen3-VL)")

    Returns:
        List of strategy dicts for the specified parser
    """
    chunking_data = get_chunking_data(data)

    if parser in chunking_data:
        return chunking_data[parser].get("strategies", [])

    # Legacy fallback
    if "_legacy" in chunking_data:
        return chunking_data["_legacy"].get("strategies", [])

    return []


def get_chunking_parsers(data: Dict[str, Any]) -> List[str]:
    """
    Get list of parsers that have chunking results.

    Returns:
        List of parser names with chunking data
    """
    chunking_data = get_chunking_data(data)

    if "_legacy" in chunking_data:
        return ["_legacy"]

    return list(chunking_data.keys())


def get_parser_names(data: Dict[str, Any]) -> List[str]:
    """Get list of parser names from config or results."""
    if "error" in data:
        return []

    # Try from config first
    if "test_config" in data and "parsers" in data["test_config"]:
        return data["test_config"]["parsers"]

    # Extract from results
    parsers = set()
    for item in data.get("parsing_results", []):
        parsers.update(item.get("results", {}).keys())
    return sorted(list(parsers))


# =============================================================================
# DataFrame Generators
# =============================================================================

def get_parsing_summary_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate summary DataFrame for all parsing tests.

    Columns: Test, Parser, WER, CER, BLEU, Latency
    """
    rows = []
    parsing_data = get_parsing_data(data)

    for test_id, test_info in parsing_data.items():
        for parser, metrics in test_info["parsers"].items():
            rows.append({
                "Test": test_info["id"],
                "Test Name": test_info["name"],
                "Parser": parser,
                "WER": metrics.get("wer", 0),
                "CER": metrics.get("cer", 0),
                "BLEU": metrics.get("bleu", 0),
                "Latency (ms)": metrics.get("latency", 0),
            })

    return pd.DataFrame(rows)


def get_chunking_summary_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate summary DataFrame for chunking strategies.

    Columns: Strategy, Params, Chunks, Mean BC, Mean CS
    """
    rows = []
    chunking_data = get_chunking_data(data)

    for strategy, info in chunking_data.items():
        params_str = ", ".join(f"{k}={v}" for k, v in info["params"].items())
        rows.append({
            "Strategy": strategy,
            "Parameters": params_str,
            "Chunks": len(info["chunks"]),
            "Mean BC": info["mean_bc"],
            "Mean CS": info["mean_cs"],
        })

    return pd.DataFrame(rows)


def get_aggregated_parser_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Get aggregated metrics per parser (mean across all tests).

    Columns: Parser, Avg WER, Avg CER, Avg BLEU, Avg Latency, Win Count
    """
    parsing_df = get_parsing_summary_df(data)

    if parsing_df.empty:
        return pd.DataFrame()

    # Aggregate by parser
    agg_df = parsing_df.groupby("Parser").agg({
        "WER": "mean",
        "CER": "mean",
        "BLEU": "mean",
        "Latency (ms)": "mean",
    }).reset_index()

    agg_df.columns = ["Parser", "Avg WER", "Avg CER", "Avg BLEU", "Avg Latency (ms)"]

    # Calculate win count (best WER per test)
    win_counts = {}
    for test in parsing_df["Test"].unique():
        test_df = parsing_df[parsing_df["Test"] == test]
        if not test_df.empty:
            best_parser = test_df.loc[test_df["WER"].idxmin(), "Parser"]
            win_counts[best_parser] = win_counts.get(best_parser, 0) + 1

    agg_df["Win Count"] = agg_df["Parser"].map(lambda p: win_counts.get(p, 0))

    return agg_df


# =============================================================================
# Pagination Helper
# =============================================================================

def paginate_data(data: List, page: int, page_size: int = 10) -> tuple:
    """
    Paginate a list of data.

    Args:
        data: List to paginate
        page: Current page (1-indexed)
        page_size: Items per page

    Returns:
        (paginated_data, total_pages, has_prev, has_next)
    """
    total = len(data)
    total_pages = (total + page_size - 1) // page_size

    if total_pages == 0:
        return [], 0, False, False

    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    return (
        data[start:end],
        total_pages,
        page > 1,
        page < total_pages
    )


# =============================================================================
# Export Helpers
# =============================================================================

def export_df_to_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV string for download."""
    return df.to_csv(index=False, encoding="utf-8-sig")


def get_chart_download_config(filename: str = "chart") -> Dict[str, Any]:
    """
    Get Plotly chart config for PNG download button.

    Usage:
        fig.update_layout(...)
        st.plotly_chart(fig, config=get_chart_download_config("my_chart"))
    """
    return {
        "toImageButtonOptions": {
            "format": "png",
            "filename": filename,
            "height": 600,
            "width": 1200,
            "scale": 2,
        },
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["downloadImage"],
    }


# =============================================================================
# Sample Data (Fallback)
# =============================================================================

def get_sample_data() -> Dict[str, Any]:
    """
    Return sample data for demonstration when no JSON file exists.

    This provides a reasonable default for development/testing.
    """
    return {
        "version": "1.0",
        "created_at": "2026-01-27T00:00:00Z",
        "test_config": {
            "parsers": ["VLM (Qwen3-VL)", "pdfplumber", "Docling (RapidOCR)"]
        },
        "parsing_results": [
            {
                "test_id": "test_1",
                "name": "Sample Document",
                "doc_type": "PDF",
                "results": {
                    "VLM (Qwen3-VL)": {"wer": 0.12, "cer": 0.09, "bleu": 0.85, "latency": 5000},
                    "pdfplumber": {"wer": 0.18, "cer": 0.14, "bleu": 0.72, "latency": 1200},
                    "Docling (RapidOCR)": {"wer": 0.25, "cer": 0.20, "bleu": 0.65, "latency": 3500},
                }
            }
        ],
        "chunking_results": [
            {
                "strategy": "Fixed",
                "params": {"chunk_size": 512, "overlap": 50},
                "chunks": [{"id": 1, "bc": 0.7, "cs": 0.8, "length": 512}],
                "mean_bc": 0.7,
                "mean_cs": 0.8,
            }
        ]
    }
