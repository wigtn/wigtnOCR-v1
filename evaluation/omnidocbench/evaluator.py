"""OmniDocBench evaluation orchestrator.

Internalized evaluation pipeline following OmniDocBench protocol.
Computes NED (text), TEDS (tables), Reading Order NED, and Formula NED.

Usage:
    evaluator = OmniDocBenchEvaluator(gt_path, pred_dir)
    results = evaluator.evaluate()

References:
- OmniDocBench (Ouyang et al., CVPR 2025)
- TEDS (Zhong et al., 2019)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

import Levenshtein

from evaluation.metrics.ned import calculate_ned
from evaluation.metrics.teds import TEDS, calculate_teds
from evaluation.omnidocbench.extractor import (
    extract_elements_from_markdown,
    extract_gt_elements,
)
from evaluation.omnidocbench.matcher import (
    match_text_elements,
    match_formula_elements,
    compute_reading_order_ned,
)
from evaluation.omnidocbench.normalizer import normalize_html_table


@dataclass
class PageResult:
    """Evaluation result for a single page."""
    image_id: str
    text_ned: float | None = None          # NED for text blocks
    table_teds: float | None = None        # TEDS for tables
    table_teds_s: float | None = None      # TEDS structure-only
    formula_ned: float | None = None       # NED for formulas
    reading_order_ned: float | None = None # Reading order NED
    n_text_matches: int = 0
    n_table_matches: int = 0
    n_formula_matches: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregated evaluation result across all pages."""
    # Text NED (lower = better)
    text_ned_page_avg: float = 0.0
    text_ned_whole: float = 0.0
    text_ned_sample_avg: float = 0.0
    # Table TEDS (higher = better)
    table_teds_avg: float = 0.0
    table_teds_s_avg: float = 0.0
    # Formula NED (lower = better)
    formula_ned_avg: float = 0.0
    # Reading Order (lower = better)
    reading_order_ned_avg: float = 0.0
    # Counts
    n_pages_evaluated: int = 0
    n_pages_skipped: int = 0
    n_text_samples: int = 0
    n_table_samples: int = 0
    n_formula_samples: int = 0
    # Per-page details
    page_results: list[PageResult] = field(default_factory=list)
    # Grouped by attributes
    group_results: dict = field(default_factory=dict)


class OmniDocBenchEvaluator:
    """Internalized OmniDocBench evaluation pipeline.

    Evaluates VLM predictions against OmniDocBench ground truth
    using the official protocol: NED (text), TEDS (tables),
    NED (formulas), NED (reading order).

    Args:
        gt_path: Path to OmniDocBench.json
        pred_dir: Directory containing prediction .md files
        images_dir: Optional directory with source images (for reference)
    """

    def __init__(
        self,
        gt_path: Path | str,
        pred_dir: Path | str,
        images_dir: Path | str | None = None,
    ):
        self.gt_path = Path(gt_path)
        self.pred_dir = Path(pred_dir)
        self.images_dir = Path(images_dir) if images_dir else None

        # Load GT
        with open(self.gt_path, "r", encoding="utf-8") as f:
            self.gt_data = json.load(f)

        # TEDS evaluators
        self._teds = TEDS(structure_only=False)
        self._teds_s = TEDS(structure_only=True)

    def evaluate(self, verbose: bool = True) -> EvalResult:
        """Run full evaluation pipeline.

        Returns:
            EvalResult with aggregated and per-page metrics.
        """
        result = EvalResult()

        # Accumulators for corpus-level NED
        all_text_edits = 0
        all_text_upper = 0
        text_sample_neds = []
        page_text_neds = []
        table_teds_scores = []
        table_teds_s_scores = []
        formula_neds = []
        reading_order_neds = []

        for page_data in self.gt_data:
            page_info = page_data.get("page_info", {})
            image_path = page_info.get("image_path", "")
            image_name = Path(image_path).stem if image_path else ""

            # Find prediction file
            pred_path = self._find_pred_file(image_name)
            if pred_path is None:
                result.n_pages_skipped += 1
                continue

            pred_content = pred_path.read_text(encoding="utf-8")
            if not pred_content or pred_content.startswith("<!-- ERROR:"):
                result.n_pages_skipped += 1
                continue

            # Extract elements
            gt_elements = extract_gt_elements(page_data)
            pred_elements = extract_elements_from_markdown(pred_content)

            page_result = PageResult(
                image_id=image_name,
                metadata=page_info.get("page_attribute", {}),
            )

            # --- Text evaluation ---
            text_matches = match_text_elements(
                gt_elements["text"], pred_elements["text"],
            )
            matched_texts = [m for m in text_matches if m["matched"]]
            page_result.n_text_matches = len(matched_texts)

            if matched_texts:
                page_edits = 0
                page_upper = 0
                for m in matched_texts:
                    norm_gt = m["norm_gt"]
                    norm_pred = m["norm_pred"]
                    upper = max(len(norm_gt), len(norm_pred))
                    if upper > 0:
                        edit = Levenshtein.distance(norm_gt, norm_pred)
                        page_edits += edit
                        page_upper += upper
                        all_text_edits += edit
                        all_text_upper += upper
                        text_sample_neds.append(edit / upper)

                page_text_ned = page_edits / page_upper if page_upper > 0 else 0.0
                page_text_neds.append(page_text_ned)
                page_result.text_ned = page_text_ned

            # --- Table evaluation ---
            gt_tables = gt_elements["table"]
            pred_tables = pred_elements["html_table"]

            if gt_tables and pred_tables:
                for gt_table in gt_tables:
                    gt_html = gt_table.html or ""
                    if not gt_html:
                        continue

                    gt_html_norm = normalize_html_table(gt_html)

                    # Find best matching pred table
                    best_teds = 0.0
                    best_teds_s = 0.0
                    for pred_table in pred_tables:
                        pred_html = normalize_html_table(pred_table.content)
                        try:
                            score = self._teds.evaluate(pred_html, gt_html_norm)
                            score_s = self._teds_s.evaluate(pred_html, gt_html_norm)
                        except Exception as e:
                            logger.debug("TEDS evaluation failed: %s", e)
                            score, score_s = 0.0, 0.0
                        if score > best_teds:
                            best_teds = score
                            best_teds_s = score_s

                    table_teds_scores.append(best_teds)
                    table_teds_s_scores.append(best_teds_s)
                    page_result.n_table_matches += 1

                page_result.table_teds = (
                    sum(table_teds_scores[-page_result.n_table_matches:])
                    / page_result.n_table_matches
                    if page_result.n_table_matches > 0 else None
                )
                page_result.table_teds_s = (
                    sum(table_teds_s_scores[-page_result.n_table_matches:])
                    / page_result.n_table_matches
                    if page_result.n_table_matches > 0 else None
                )

            # --- Formula evaluation ---
            formula_matches = match_formula_elements(
                gt_elements["formula"], pred_elements["formula"],
            )
            matched_formulas = [m for m in formula_matches if m["matched"]]
            page_result.n_formula_matches = len(matched_formulas)

            if matched_formulas:
                page_formula_neds = [m["edit"] for m in matched_formulas]
                page_result.formula_ned = sum(page_formula_neds) / len(page_formula_neds)
                formula_neds.extend(page_formula_neds)

            # --- Reading order evaluation ---
            # Use text+formula matches combined
            all_matches = text_matches + formula_matches
            all_gt_items = gt_elements.get("reading_order_items", [])
            if len(all_matches) > 1 and all_gt_items:
                ro_ned = compute_reading_order_ned(text_matches, gt_elements["text"])
                page_result.reading_order_ned = ro_ned
                reading_order_neds.append(ro_ned)

            result.page_results.append(page_result)
            result.n_pages_evaluated += 1

            if verbose and result.n_pages_evaluated % 100 == 0:
                print(f"  Evaluated {result.n_pages_evaluated} pages...")

        # Aggregate
        result.text_ned_whole = all_text_edits / all_text_upper if all_text_upper > 0 else 0.0
        result.text_ned_sample_avg = (
            sum(text_sample_neds) / len(text_sample_neds) if text_sample_neds else 0.0
        )
        result.text_ned_page_avg = (
            sum(page_text_neds) / len(page_text_neds) if page_text_neds else 0.0
        )
        result.table_teds_avg = (
            sum(table_teds_scores) / len(table_teds_scores) if table_teds_scores else 0.0
        )
        result.table_teds_s_avg = (
            sum(table_teds_s_scores) / len(table_teds_s_scores) if table_teds_s_scores else 0.0
        )
        result.formula_ned_avg = (
            sum(formula_neds) / len(formula_neds) if formula_neds else 0.0
        )
        result.reading_order_ned_avg = (
            sum(reading_order_neds) / len(reading_order_neds) if reading_order_neds else 0.0
        )
        result.n_text_samples = len(text_sample_neds)
        result.n_table_samples = len(table_teds_scores)
        result.n_formula_samples = len(formula_neds)

        return result

    def _find_pred_file(self, image_name: str) -> Path | None:
        """Find prediction .md file matching image name."""
        # Try common patterns
        candidates = [
            self.pred_dir / f"{image_name}.md",
            self.pred_dir / f"{image_name.replace('.pdf', '')}.md",
            self.pred_dir / f"{image_name}.mmd",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def print_summary(self, result: EvalResult) -> None:
        """Print formatted summary of evaluation results."""
        print("=" * 65)
        print(f"  OmniDocBench Evaluation Results")
        print(f"  Predictions: {self.pred_dir}")
        print("=" * 65)
        print(f"  Pages evaluated: {result.n_pages_evaluated}")
        print(f"  Pages skipped:   {result.n_pages_skipped}")
        print()
        print(f"  Text NED (page_avg):       {result.text_ned_page_avg:.4f}  (lower=better)")
        print(f"  Text NED (edit_whole):      {result.text_ned_whole:.4f}")
        print(f"  Text NED (sample_avg):      {result.text_ned_sample_avg:.4f}")
        print(f"  Text samples:               {result.n_text_samples}")
        print()
        print(f"  Table TEDS:                 {result.table_teds_avg:.4f}  (higher=better)")
        print(f"  Table TEDS-S:               {result.table_teds_s_avg:.4f}")
        print(f"  Table samples:              {result.n_table_samples}")
        print()
        print(f"  Formula NED:                {result.formula_ned_avg:.4f}  (lower=better)")
        print(f"  Formula samples:            {result.n_formula_samples}")
        print()
        print(f"  Reading Order NED:          {result.reading_order_ned_avg:.4f}  (lower=better)")
        print("=" * 65)

    def to_dict(self, result: EvalResult) -> dict:
        """Convert EvalResult to serializable dict."""
        return {
            "text": {
                "ned_page_avg": result.text_ned_page_avg,
                "ned_whole": result.text_ned_whole,
                "ned_sample_avg": result.text_ned_sample_avg,
                "n_samples": result.n_text_samples,
            },
            "table": {
                "teds": result.table_teds_avg,
                "teds_s": result.table_teds_s_avg,
                "n_samples": result.n_table_samples,
            },
            "formula": {
                "ned_avg": result.formula_ned_avg,
                "n_samples": result.n_formula_samples,
            },
            "reading_order": {
                "ned_avg": result.reading_order_ned_avg,
            },
            "summary": {
                "n_pages_evaluated": result.n_pages_evaluated,
                "n_pages_skipped": result.n_pages_skipped,
            },
        }
