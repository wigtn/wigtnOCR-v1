"""
Data loading and transformation utilities for the dashboard.

Supports:
1. Manual input of test results
2. Loading from benchmark JSON files
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParserResult:
    """Single parser result for a test."""
    wer: float = 0.0
    cer: float = 0.0
    bleu: float = 0.0
    char_acc: float = 0.0
    latency: float = 0.0
    success: bool = True
    content_length: int = 0


@dataclass
class ChunkingMetrics:
    """Chunking quality metrics."""
    bc: float = 0.0  # Boundary Clarity (higher is better)
    cs: float = 0.0  # Chunk Stickiness (lower is better)


@dataclass
class TestResult:
    """Complete result for a single test case."""
    name: str
    doc_type: str = "unknown"
    parsers: Dict[str, ParserResult] = field(default_factory=dict)
    chunking: Optional[ChunkingMetrics] = None


# =============================================================================
# Data Loader Class
# =============================================================================

class DataLoader:
    """Load and transform test data from various sources."""

    def __init__(self):
        self.test_results: Dict[str, TestResult] = {}
        self._raw_data: Dict[str, Any] = {}

    def load_from_manual_input(self, data: Dict[str, Any]) -> None:
        """Load data from manual input dictionary.

        Expected format:
        {
            "test_1": {
                "name": "Paper PDF",
                "doc_type": "paper",
                "parsers": {
                    "vlm": {"wer": 0.15, "bleu": 0.82, ...},
                    "ocr": {"wer": 0.25, "bleu": 0.65, ...},
                },
                "chunking": {"bc": 0.95, "cs": 1.2}
            },
            ...
        }
        """
        self._raw_data = data
        self.test_results = {}

        for test_id, test_data in data.items():
            parsers = {}
            for parser_name, parser_data in test_data.get("parsers", {}).items():
                parsers[parser_name] = ParserResult(
                    wer=parser_data.get("wer", 0.0),
                    cer=parser_data.get("cer", 0.0),
                    bleu=parser_data.get("bleu", 0.0),
                    char_acc=parser_data.get("char_acc", 0.0),
                    latency=parser_data.get("latency", 0.0),
                    success=parser_data.get("success", True),
                    content_length=parser_data.get("content_length", 0),
                )

            chunking = None
            if "chunking" in test_data:
                chunking = ChunkingMetrics(
                    bc=test_data["chunking"].get("bc", 0.0),
                    cs=test_data["chunking"].get("cs", 0.0),
                )

            self.test_results[test_id] = TestResult(
                name=test_data.get("name", test_id),
                doc_type=test_data.get("doc_type", "unknown"),
                parsers=parsers,
                chunking=chunking,
            )

    def load_from_json_file(self, file_path: Path) -> None:
        """Load data from a benchmark JSON file.

        Expected format (from run_benchmark.py output):
        {
            "benchmark_info": {...},
            "results": {
                "test_1": {
                    "parsers": {...},
                    "chunking": {...}
                }
            }
        }
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._raw_data = data

        # Handle both direct format and nested format
        if "results" in data:
            results_data = data["results"]
        else:
            results_data = data

        self.load_from_manual_input(results_data)

    def load_from_json_string(self, json_string: str) -> None:
        """Load data from a JSON string."""
        data = json.loads(json_string)

        if "results" in data:
            results_data = data["results"]
        else:
            results_data = data

        self.load_from_manual_input(results_data)

    # =========================================================================
    # Data Transformation Methods
    # =========================================================================

    def get_parser_metrics_df(
        self,
        test_ids: Optional[List[str]] = None,
        parsers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get parser performance metrics as a DataFrame.

        Returns DataFrame with columns:
        - test_id, test_name, parser, wer, cer, bleu, char_acc, latency
        """
        rows = []

        for test_id, result in self.test_results.items():
            if test_ids and test_id not in test_ids:
                continue

            for parser_name, parser_result in result.parsers.items():
                if parsers and parser_name.lower() not in [p.lower() for p in parsers]:
                    continue

                rows.append({
                    "test_id": test_id,
                    "test_name": result.name,
                    "doc_type": result.doc_type,
                    "parser": parser_name,
                    "wer": parser_result.wer,
                    "cer": parser_result.cer,
                    "bleu": parser_result.bleu,
                    "char_acc": parser_result.char_acc,
                    "latency": parser_result.latency,
                    "success": parser_result.success,
                })

        return pd.DataFrame(rows)

    def get_chunking_metrics_df(
        self,
        test_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get chunking metrics as a DataFrame.

        Returns DataFrame with columns:
        - test_id, test_name, bc, cs
        """
        rows = []

        for test_id, result in self.test_results.items():
            if test_ids and test_id not in test_ids:
                continue

            if result.chunking:
                rows.append({
                    "test_id": test_id,
                    "test_name": result.name,
                    "doc_type": result.doc_type,
                    "bc": result.chunking.bc,
                    "cs": result.chunking.cs,
                })

        return pd.DataFrame(rows)

    def get_aggregated_metrics(
        self,
        parsers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get aggregated metrics per parser (mean across all tests).

        Returns DataFrame with columns:
        - parser, avg_wer, avg_cer, avg_bleu, avg_char_acc, avg_latency, win_count
        """
        df = self.get_parser_metrics_df(parsers=parsers)

        if df.empty:
            return pd.DataFrame()

        # Calculate aggregates
        agg_df = df.groupby("parser").agg({
            "wer": "mean",
            "cer": "mean",
            "bleu": "mean",
            "char_acc": "mean",
            "latency": "mean",
        }).reset_index()

        agg_df.columns = ["parser", "avg_wer", "avg_cer", "avg_bleu", "avg_char_acc", "avg_latency"]

        # Calculate win count (best WER per test)
        win_counts = {}
        for test_id in df["test_id"].unique():
            test_df = df[df["test_id"] == test_id]
            if not test_df.empty:
                best_parser = test_df.loc[test_df["wer"].idxmin(), "parser"]
                win_counts[best_parser] = win_counts.get(best_parser, 0) + 1

        agg_df["win_count"] = agg_df["parser"].map(lambda p: win_counts.get(p, 0))

        return agg_df

    def get_heatmap_data(
        self,
        metric: str = "wer",
        test_ids: Optional[List[str]] = None,
        parsers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Get data formatted for heatmap visualization.

        Returns pivot table with:
        - Rows: doc_type or test_name
        - Columns: parsers
        - Values: specified metric
        """
        df = self.get_parser_metrics_df(test_ids=test_ids, parsers=parsers)

        if df.empty:
            return pd.DataFrame()

        # Pivot table
        pivot = df.pivot_table(
            index="test_name",
            columns="parser",
            values=metric,
            aggfunc="mean"
        )

        return pivot

    # =========================================================================
    # KPI Calculation Methods
    # =========================================================================

    def get_best_parser(self) -> tuple:
        """Get the best parser based on win rate.

        Returns:
            (parser_name, win_rate)
        """
        agg_df = self.get_aggregated_metrics()

        if agg_df.empty:
            return ("N/A", 0.0)

        total_tests = len(self.test_results)
        if total_tests == 0:
            return ("N/A", 0.0)

        best_row = agg_df.loc[agg_df["win_count"].idxmax()]
        win_rate = best_row["win_count"] / total_tests

        return (best_row["parser"], win_rate)

    def get_average_wer(self, parser: Optional[str] = None) -> float:
        """Get average WER across all tests or for a specific parser."""
        df = self.get_parser_metrics_df()

        if df.empty:
            return 0.0

        if parser:
            df = df[df["parser"].str.lower() == parser.lower()]

        return df["wer"].mean() if not df.empty else 0.0

    def get_test_coverage(self) -> tuple:
        """Get test coverage statistics.

        Returns:
            (completed_tests, total_tests)
        """
        total = len(self.test_results)
        completed = sum(
            1 for r in self.test_results.values()
            if any(p.success for p in r.parsers.values())
        )
        return (completed, total)

    def get_test_ids(self) -> List[str]:
        """Get list of all test IDs."""
        return list(self.test_results.keys())

    def get_parser_names(self) -> List[str]:
        """Get list of all unique parser names."""
        parsers = set()
        for result in self.test_results.values():
            parsers.update(result.parsers.keys())
        return sorted(list(parsers))


# =============================================================================
# Sample Data Generator (for demo/testing)
# =============================================================================

def generate_sample_data() -> Dict[str, Any]:
    """Generate sample test data for demonstration."""
    return {
        "test_1": {
            "name": "Academic Paper PDF",
            "doc_type": "paper",
            "parsers": {
                "VLM": {"wer": 0.12, "cer": 0.08, "bleu": 0.85, "char_acc": 0.96, "latency": 5.2},
                "OCR": {"wer": 0.25, "cer": 0.18, "bleu": 0.68, "char_acc": 0.88, "latency": 1.5},
                "Docling": {"wer": 0.18, "cer": 0.12, "bleu": 0.78, "char_acc": 0.92, "latency": 3.8},
            },
            "chunking": {"bc": 0.92, "cs": 1.1},
        },
        "test_2": {
            "name": "Invoice Scan",
            "doc_type": "invoice",
            "parsers": {
                "VLM": {"wer": 0.08, "cer": 0.05, "bleu": 0.92, "char_acc": 0.98, "latency": 4.8},
                "OCR": {"wer": 0.35, "cer": 0.28, "bleu": 0.55, "char_acc": 0.78, "latency": 1.2},
                "Docling": {"wer": 0.22, "cer": 0.15, "bleu": 0.72, "char_acc": 0.88, "latency": 3.5},
            },
            "chunking": {"bc": 0.88, "cs": 1.4},
        },
        "test_3": {
            "name": "Technical Manual",
            "doc_type": "manual",
            "parsers": {
                "VLM": {"wer": 0.15, "cer": 0.10, "bleu": 0.82, "char_acc": 0.94, "latency": 6.1},
                "OCR": {"wer": 0.20, "cer": 0.14, "bleu": 0.75, "char_acc": 0.90, "latency": 1.8},
                "Docling": {"wer": 0.16, "cer": 0.11, "bleu": 0.80, "char_acc": 0.93, "latency": 4.2},
            },
            "chunking": {"bc": 0.95, "cs": 0.9},
        },
    }


def load_benchmark_files(results_dir: Path) -> List[Path]:
    """Find all benchmark JSON files in the results directory."""
    if not results_dir.exists():
        return []

    return sorted(results_dir.glob("benchmark_*.json"))
