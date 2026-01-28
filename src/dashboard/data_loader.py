"""
Data loading utilities for VLM Document Parsing Dashboard.

Supports loading test results from results/test_*/ folders.

Usage:
    from dashboard.data_loader import load_all_results, load_all_results_cached

    # With Streamlit caching
    data = load_all_results_cached()

    # Without caching (for CLI usage)
    data = load_all_results()
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
CACHE_TTL = 300  # 5 minutes (shorter for development)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TestResult:
    """Single test result with parsing and chunking data."""
    test_id: str
    folder_path: Path
    evaluation: Dict[str, Any]  # From evaluation.json
    chunking: Optional[Dict[str, Any]]  # From chunking.json

    @property
    def has_chunking(self) -> bool:
        return self.chunking is not None and len(self.chunking.get("results", {})) > 0

    @property
    def parsers(self) -> List[str]:
        # Support both "parsers" (new format) and "results" (legacy format)
        parser_results = self.evaluation.get("parsers", {}) or self.evaluation.get("results", {})
        return list(parser_results.keys())


# =============================================================================
# Folder Scanning
# =============================================================================

def scan_test_folders(results_dir: Path = None) -> List[Path]:
    """
    Scan results/ folder for all test_* directories.

    Returns:
        Sorted list of test folder paths
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR

    if not results_dir.exists():
        return []

    folders = []
    for folder in sorted(results_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("test_"):
            # Check for evaluation.json
            if (folder / "evaluation.json").exists():
                folders.append(folder)

    return folders


def load_test_result(folder: Path) -> Optional[TestResult]:
    """
    Load a single test result from a folder.

    Reads:
        - evaluation.json (parsing results)
        - chunking.json (chunking results, optional)

    Returns:
        TestResult or None if folder is invalid
    """
    eval_file = folder / "evaluation.json"
    chunk_file = folder / "chunking.json"

    if not eval_file.exists():
        return None

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            evaluation = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    chunking = None
    if chunk_file.exists():
        try:
            with open(chunk_file, "r", encoding="utf-8") as f:
                chunking = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return TestResult(
        test_id=folder.name,
        folder_path=folder,
        evaluation=evaluation,
        chunking=chunking
    )


def load_all_results(results_dir: Path = None) -> Dict[str, Any]:
    """
    Load all test results from results/ folder.

    Returns:
        Dict with structure:
        {
            "version": "2.0",
            "source": "folder_scan",
            "results_dir": str,
            "tests": {
                "test_1": {
                    "test_id": "test_1",
                    "source_file": str,
                    "evaluation": {...},
                    "chunking": {...}
                },
                ...
            },
            "parsers": ["VLM", "OCR-Text", ...],
            "test_count": int,
            "has_chunking": bool
        }
    """
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    folders = scan_test_folders(results_dir)

    if not folders:
        return {
            "error": f"테스트 결과가 없습니다: {results_dir}\n테스트를 먼저 실행하세요: python -m src.eval_parsers --all",
            "results_dir": str(results_dir),
        }

    tests = {}
    all_parsers = set()
    has_any_chunking = False

    for folder in folders:
        result = load_test_result(folder)
        if result:
            tests[result.test_id] = {
                "test_id": result.test_id,
                "source_file": str(result.evaluation.get("pdf", "")),
                "timestamp": result.evaluation.get("timestamp", ""),
                "evaluation": result.evaluation,
                "chunking": result.chunking,
            }
            all_parsers.update(result.parsers)
            if result.has_chunking:
                has_any_chunking = True

    return {
        "version": "2.0",
        "source": "folder_scan",
        "results_dir": str(results_dir),
        "tests": tests,
        "parsers": sorted(list(all_parsers)),
        "test_count": len(tests),
        "has_chunking": has_any_chunking,
        "loaded_at": datetime.now().isoformat(),
    }


def load_all_results_cached(results_dir: Path = None) -> Dict[str, Any]:
    """
    Load all results with Streamlit caching.
    """
    if not HAS_STREAMLIT:
        return load_all_results(results_dir)

    @st.cache_data(ttl=CACHE_TTL, show_spinner=False)
    def _cached_load(dir_str: str) -> Dict[str, Any]:
        return load_all_results(Path(dir_str))

    results_dir = results_dir or DEFAULT_RESULTS_DIR
    return _cached_load(str(results_dir))


# =============================================================================
# Data Transformation Functions
# =============================================================================

def get_test_ids(data: Dict[str, Any]) -> List[str]:
    """Get sorted list of test IDs."""
    if "error" in data:
        return []
    return sorted(data.get("tests", {}).keys())


def get_parser_names(data: Dict[str, Any]) -> List[str]:
    """Get list of all parser names."""
    if "error" in data:
        return []
    return data.get("parsers", [])


def get_test_evaluation(data: Dict[str, Any], test_id: str) -> Dict[str, Any]:
    """Get evaluation data for a specific test."""
    if "error" in data:
        return {}
    return data.get("tests", {}).get(test_id, {}).get("evaluation", {})


def get_test_chunking(data: Dict[str, Any], test_id: str) -> Optional[Dict[str, Any]]:
    """Get chunking data for a specific test."""
    if "error" in data:
        return None
    return data.get("tests", {}).get(test_id, {}).get("chunking")


# =============================================================================
# DataFrame Generators
# =============================================================================

def get_parsing_summary_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate summary DataFrame for all parsing tests.

    Columns: Test ID, Source, Parser, CER, WER, Structure F1, Latency, Status
    """
    if "error" in data:
        return pd.DataFrame()

    rows = []
    for test_id, test_data in data.get("tests", {}).items():
        eval_data = test_data.get("evaluation", {})
        source_file = eval_data.get("pdf", "")

        # Support both "parsers" (new format) and "results" (legacy format)
        parser_results = eval_data.get("parsers", {}) or eval_data.get("results", {})
        for parser, metrics in parser_results.items():
            cer = metrics.get("cer")
            wer = metrics.get("wer")
            structure_f1 = metrics.get("structure_f1")

            rows.append({
                "Test ID": test_id,
                "Source": Path(source_file).name if source_file else "",
                "Parser": parser,
                "CER": cer if cer is not None else float("nan"),
                "WER": wer if wer is not None else float("nan"),
                "CER %": f"{cer*100:.1f}%" if cer is not None else "N/A",
                "WER %": f"{wer*100:.1f}%" if wer is not None else "N/A",
                "Structure F1": structure_f1 if structure_f1 is not None else float("nan"),
                "Struct-F1 %": f"{structure_f1*100:.1f}%" if structure_f1 is not None else "N/A",
                "Latency (s)": metrics.get("elapsed_time", 0),
                "Content Length": metrics.get("content_length", 0),
                "Success": "✓" if metrics.get("success", False) else "✗",
            })

    return pd.DataFrame(rows)


def get_chunking_summary_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate summary DataFrame for all chunking tests.

    Columns: Test ID, Parser, BC Score, CS Score, Chunk Count
    """
    if "error" in data:
        return pd.DataFrame()

    rows = []
    for test_id, test_data in data.get("tests", {}).items():
        chunking = test_data.get("chunking")
        if not chunking:
            continue

        for parser, result in chunking.get("results", {}).items():
            bc = result.get("bc", {})
            cs = result.get("cs", {})

            rows.append({
                "Test ID": test_id,
                "Parser": parser,
                "BC Score": bc.get("score") if bc else None,
                "BC Min": bc.get("min") if bc else None,
                "BC Max": bc.get("max") if bc else None,
                "CS Score": cs.get("score") if cs else None,
                "Chunk Count": result.get("chunk_count", 0),
            })

    return pd.DataFrame(rows)


def get_aggregated_parser_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Get aggregated metrics per parser (mean across all tests).

    Columns: Parser, Avg CER, Avg WER, Test Count, Best CER Count
    """
    parsing_df = get_parsing_summary_df(data)

    if parsing_df.empty:
        return pd.DataFrame()

    # Filter to only successful parses with valid CER
    valid_df = parsing_df[parsing_df["Success"] == "✓"].dropna(subset=["CER"])

    if valid_df.empty:
        return pd.DataFrame()

    # Aggregate by parser
    agg_df = valid_df.groupby("Parser").agg({
        "CER": ["mean", "std", "count"],
        "WER": ["mean", "std"],
        "Latency (s)": "mean",
    }).reset_index()

    # Flatten column names
    agg_df.columns = ["Parser", "Avg CER", "CER Std", "Test Count", "Avg WER", "WER Std", "Avg Latency"]

    # Calculate win count (best CER per test)
    win_counts = {}
    for test_id in valid_df["Test ID"].unique():
        test_df = valid_df[valid_df["Test ID"] == test_id]
        if not test_df.empty:
            best_parser = test_df.loc[test_df["CER"].idxmin(), "Parser"]
            win_counts[best_parser] = win_counts.get(best_parser, 0) + 1

    agg_df["Best CER Count"] = agg_df["Parser"].map(lambda p: win_counts.get(p, 0))

    return agg_df


def get_test_detail_df(data: Dict[str, Any], test_id: str) -> pd.DataFrame:
    """
    Get detailed parsing results for a specific test.
    """
    eval_data = get_test_evaluation(data, test_id)
    if not eval_data:
        return pd.DataFrame()

    rows = []
    # Support both "parsers" (new format) and "results" (legacy format)
    parser_results = eval_data.get("parsers", {}) or eval_data.get("results", {})
    for parser, metrics in parser_results.items():
        rows.append({
            "Parser": parser,
            "Success": metrics.get("success", False),
            "CER": metrics.get("cer"),
            "WER": metrics.get("wer"),
            "Structure F1": metrics.get("structure_f1"),
            "Latency (s)": metrics.get("elapsed_time", 0),
            "Content Length": metrics.get("content_length", 0),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Chunking Data Helpers
# =============================================================================

def get_chunking_for_test(data: Dict[str, Any], test_id: str) -> Dict[str, Any]:
    """
    Get formatted chunking data for a specific test.

    Returns:
        {
            "config": {...},
            "parsers": {
                "VLM": {"bc": {...}, "cs": {...}, "chunks": [...]},
                ...
            }
        }
    """
    chunking = get_test_chunking(data, test_id)
    if not chunking:
        return {}

    return {
        "config": chunking.get("config", {}),
        "timestamp": chunking.get("timestamp", ""),
        "parsers": chunking.get("results", {}),
    }


def get_tests_with_chunking(data: Dict[str, Any]) -> List[str]:
    """Get list of test IDs that have chunking results."""
    if "error" in data:
        return []

    tests_with_chunking = []
    for test_id, test_data in data.get("tests", {}).items():
        if test_data.get("chunking"):
            tests_with_chunking.append(test_id)

    return sorted(tests_with_chunking)


# =============================================================================
# Export Helpers
# =============================================================================

def export_df_to_csv(df: pd.DataFrame) -> str:
    """Export DataFrame to CSV string for download."""
    return df.to_csv(index=False, encoding="utf-8-sig")


def get_chart_download_config(filename: str = "chart") -> Dict[str, Any]:
    """
    Get Plotly chart config for PNG download button.
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
# Backward Compatibility (Legacy Functions)
# =============================================================================

def load_results(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    [DEPRECATED] Load from parsing_results.json.
    Use load_all_results() instead.
    """
    # If path specified, try legacy loading
    if results_path and results_path.exists() and results_path.is_file():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Fall back to new folder scanning
    return load_all_results()


def load_results_cached(results_path: Optional[Path] = None) -> Dict[str, Any]:
    """[DEPRECATED] Use load_all_results_cached() instead."""
    return load_all_results_cached()


def get_parsing_data(data: Dict[str, Any]) -> Dict[str, Dict]:
    """
    [DEPRECATED] Transform for old dashboard format.
    New dashboard should use get_parsing_summary_df().
    """
    if "error" in data:
        return {}

    # New format (v2.0)
    if data.get("version") == "2.0":
        result = {}
        for test_id, test_data in data.get("tests", {}).items():
            eval_data = test_data.get("evaluation", {})
            metadata = eval_data.get("metadata", {})

            # 메타데이터에서 정보 추출 (자동 추출 형식 지원)
            title = metadata.get("title", test_id.replace("_", " ").title())
            filename = metadata.get("filename", Path(eval_data.get("pdf", "")).name)
            doc_type = metadata.get("doc_type", "unknown")
            pages = metadata.get("pages", 0)
            file_size_kb = metadata.get("file_size_kb", 0)
            language = metadata.get("language", "unknown")
            has_text_layer = metadata.get("has_text_layer", False)

            result[test_id] = {
                "id": title,
                "name": filename,
                "doc_type": doc_type,
                "pages": pages,
                "file_size_kb": file_size_kb,
                "language": language,
                "has_text_layer": has_text_layer,
                "source_path": eval_data.get("pdf", ""),
                "parsers": eval_data.get("parsers", {}) or eval_data.get("results", {}),
                "metadata": metadata,
            }
        return result

    # Legacy format
    result = {}
    for item in data.get("parsing_results", []):
        test_id = item.get("test_id", "unknown")
        result[test_id] = {
            "id": test_id.replace("_", " ").title(),
            "name": item.get("name", test_id),
            "doc_type": item.get("doc_type", "unknown"),
            "description": "",
            "source_path": item.get("source_path", ""),
            "parsers": item.get("results", {}),
            "metadata": {},
        }
    return result


def get_chunking_data(data: Dict[str, Any]) -> Dict[str, Dict]:
    """[DEPRECATED] Transform for old dashboard format."""
    if "error" in data:
        return {}

    # New format (v2.0)
    if data.get("version") == "2.0":
        # Aggregate chunking from all tests
        all_chunking = {}
        for test_id, test_data in data.get("tests", {}).items():
            chunking = test_data.get("chunking")
            if chunking:
                for parser, result in chunking.get("results", {}).items():
                    if parser not in all_chunking:
                        all_chunking[parser] = {
                            "parser": parser,
                            "test_results": {}
                        }
                    all_chunking[parser]["test_results"][test_id] = result
        return all_chunking

    # Legacy format
    return data.get("chunking_results", {})


def paginate_data(items: List, page: int, page_size: int) -> tuple:
    """
    Paginate a list of items.

    Returns:
        Tuple of (paginated_items, total_pages, start_idx, end_idx)
    """
    total = len(items)
    total_pages = (total + page_size - 1) // page_size
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)

    return items[start_idx:end_idx], total_pages, start_idx, end_idx


def get_chunking_parsers(data: Dict[str, Any]) -> List[str]:
    """
    Get list of parsers that have chunking data.

    Returns:
        List of parser names with chunking results
    """
    if "error" in data:
        return []

    parsers_with_chunking = set()

    # New format (v2.0)
    if data.get("version") == "2.0":
        for test_id, test_data in data.get("tests", {}).items():
            chunking = test_data.get("chunking")
            if chunking:
                for parser in chunking.get("results", {}).keys():
                    parsers_with_chunking.add(parser)
        return sorted(list(parsers_with_chunking))

    # Legacy format
    chunking_data = data.get("chunking_results", {})
    return sorted(list(chunking_data.keys()))


def get_chunking_data_for_parser(data: Dict[str, Any], parser: str) -> List[Dict]:
    """
    Get chunking data for a specific parser across all tests.

    Returns:
        List of chunking strategy results for the parser
    """
    if "error" in data:
        return []

    results = []

    # New format (v2.0)
    if data.get("version") == "2.0":
        for test_id, test_data in data.get("tests", {}).items():
            chunking = test_data.get("chunking")
            if chunking and parser in chunking.get("results", {}):
                parser_result = chunking["results"][parser]
                bc_data = parser_result.get("bc", {})
                cs_data = parser_result.get("cs", {})

                # Transform to expected format
                result = {
                    "test_id": test_id,
                    "strategy": chunking.get("config", {}).get("strategy", "unknown"),
                    "mean_bc": bc_data.get("score", 0),
                    "std_bc": bc_data.get("std", 0),
                    "mean_cs": cs_data.get("score") if cs_data else None,
                    "std_cs": cs_data.get("std") if cs_data else None,
                    "chunk_count": parser_result.get("chunk_count", 0),
                    "bc_by_sentence": [],  # Not stored in new format
                }
                results.append(result)
        return results

    # Legacy format
    chunking_data = data.get("chunking_results", {})
    parser_data = chunking_data.get(parser, {})
    return parser_data.get("strategies", [])
