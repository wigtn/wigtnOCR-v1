"""Structure evaluation metrics for document parsing.

Replaces naive element-counting Structure F1 with academically grounded metrics:
- Element-level NED: Normalized Edit Distance per matched element (OmniDocBench methodology)
- Reading Order NED: Edit distance on element ordering sequence
- Element Detection F1: Precision/Recall of detected structural elements

References:
- OmniDocBench (Ouyang et al., CVPR 2025): Element-level NED + Reading Order Edit Distance
- Upstage dp-bench: NID (Normalized Indel Distance) as supplementary metric
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import Levenshtein


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StructureElement:
    """A structural element extracted from markdown."""
    element_type: str       # heading, table, list, code_block, text
    content: str            # raw text content
    order: int              # position in document (0-indexed)
    level: Optional[int] = None  # heading level (1-6) or list depth


@dataclass
class StructureEvalResult:
    """Comprehensive structure evaluation result."""
    # Element-level NED (lower is better, 0 = perfect)
    element_ned: float
    # Reading Order NED (lower is better, 0 = perfect)
    reading_order_ned: float
    # Element Detection F1 (higher is better, 1 = perfect)
    detection_f1: float
    detection_precision: float
    detection_recall: float
    # Per-type breakdown
    per_type_ned: dict[str, float] = field(default_factory=dict)
    # Counts
    pred_element_count: int = 0
    gt_element_count: int = 0
    matched_count: int = 0

    @property
    def f1(self) -> float:
        """Legacy alias for detection_f1."""
        return self.detection_f1


# Legacy alias for backward compatibility
StructureF1Result = StructureEvalResult


# ---------------------------------------------------------------------------
# Element extraction from markdown
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
_TABLE_ROW_RE = re.compile(r'^\|(.+)\|$')
_TABLE_SEP_RE = re.compile(r'^\|[\s\-:]+\|$')
_LIST_RE = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.*)')
_CODE_FENCE_RE = re.compile(r'^```')


def extract_elements(text: str) -> list[StructureElement]:
    """Extract structural elements from markdown text.

    Groups consecutive table rows and list items into single elements.
    Remaining text blocks are captured as 'text' elements.
    """
    elements: list[StructureElement] = []
    lines = text.split('\n')
    order = 0
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Heading
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            content = m.group(2).strip()
            elements.append(StructureElement(
                element_type='heading', content=content,
                order=order, level=level,
            ))
            order += 1
            i += 1
            continue

        # Code block (fenced)
        if _CODE_FENCE_RE.match(line.strip()):
            code_lines = [line]
            i += 1
            while i < len(lines):
                code_lines.append(lines[i])
                if _CODE_FENCE_RE.match(lines[i].strip()) and len(code_lines) > 1:
                    i += 1
                    break
                i += 1
            elements.append(StructureElement(
                element_type='code_block',
                content='\n'.join(code_lines),
                order=order,
            ))
            order += 1
            continue

        # Table (group consecutive rows)
        if _TABLE_ROW_RE.match(line.strip()):
            table_lines = []
            while i < len(lines):
                stripped = lines[i].strip()
                if _TABLE_ROW_RE.match(stripped) or _TABLE_SEP_RE.match(stripped):
                    if not _TABLE_SEP_RE.match(stripped):
                        table_lines.append(stripped)
                    i += 1
                else:
                    break
            if table_lines:
                elements.append(StructureElement(
                    element_type='table',
                    content='\n'.join(table_lines),
                    order=order,
                ))
                order += 1
            continue

        # List item (group consecutive)
        m = _LIST_RE.match(line)
        if m:
            list_lines = []
            while i < len(lines):
                lm = _LIST_RE.match(lines[i])
                if lm:
                    list_lines.append(lm.group(3).strip())
                    i += 1
                else:
                    break
            elements.append(StructureElement(
                element_type='list',
                content='\n'.join(list_lines),
                order=order,
            ))
            order += 1
            continue

        # Text block (non-empty, non-structural)
        if line.strip():
            text_lines = [line.strip()]
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if (not stripped
                        or _HEADING_RE.match(lines[i])
                        or _CODE_FENCE_RE.match(stripped)
                        or _TABLE_ROW_RE.match(stripped)
                        or _LIST_RE.match(lines[i])):
                    break
                text_lines.append(stripped)
                i += 1
            elements.append(StructureElement(
                element_type='text',
                content=' '.join(text_lines),
                order=order,
            ))
            order += 1
            continue

        i += 1  # skip blank lines

    return elements


# ---------------------------------------------------------------------------
# Normalized Edit Distance (NED)
# ---------------------------------------------------------------------------

def _ned(pred: str, gt: str) -> float:
    """Normalized Edit Distance: edit_dist / max(len(pred), len(gt)).

    Returns 0.0 for perfect match, 1.0 for completely different.
    Follows OmniDocBench convention.
    """
    if not pred and not gt:
        return 0.0
    upper = max(len(pred), len(gt))
    if upper == 0:
        return 0.0
    return Levenshtein.distance(pred, gt) / upper


# ---------------------------------------------------------------------------
# Hungarian matching for element pairs
# ---------------------------------------------------------------------------

def _match_elements(
    pred_elements: list[StructureElement],
    gt_elements: list[StructureElement],
) -> list[tuple[Optional[int], Optional[int], float]]:
    """Match predicted elements to GT elements using Hungarian algorithm.

    Returns list of (pred_idx, gt_idx, ned_score) tuples.
    Unmatched elements have None for the missing side.
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    n_pred = len(pred_elements)
    n_gt = len(gt_elements)

    if n_pred == 0 and n_gt == 0:
        return []
    if n_pred == 0:
        return [(None, j, 1.0) for j in range(n_gt)]
    if n_gt == 0:
        return [(i, None, 1.0) for i in range(n_pred)]

    # Build cost matrix: NED between each pred-gt pair
    # Add type mismatch penalty
    cost = np.ones((n_pred, n_gt), dtype=np.float64)
    for i, pe in enumerate(pred_elements):
        for j, ge in enumerate(gt_elements):
            text_ned = _ned(pe.content, ge.content)
            # Type match bonus: same type gets pure NED, different type gets penalty
            if pe.element_type == ge.element_type:
                cost[i, j] = text_ned
            else:
                cost[i, j] = min(1.0, text_ned + 0.2)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    matched = set()
    results = []
    for i, j in zip(row_ind, col_ind):
        # Only accept match if cost < 1.0 (otherwise it's better unmatched)
        if cost[i, j] < 1.0:
            results.append((i, j, cost[i, j]))
            matched.add(('pred', i))
            matched.add(('gt', j))
        else:
            results.append((i, None, 1.0))
            results.append((None, j, 1.0))
            matched.add(('pred', i))
            matched.add(('gt', j))

    # Add unmatched
    for i in range(n_pred):
        if ('pred', i) not in matched:
            results.append((i, None, 1.0))
    for j in range(n_gt):
        if ('gt', j) not in matched:
            results.append((None, j, 1.0))

    return results


# ---------------------------------------------------------------------------
# Reading Order evaluation
# ---------------------------------------------------------------------------

def _reading_order_ned(
    matches: list[tuple[Optional[int], Optional[int], float]],
    pred_elements: list[StructureElement],
    gt_elements: list[StructureElement],
) -> float:
    """Compute Reading Order NED from matched element pairs.

    Constructs two sequences:
    - GT order: sorted by GT element order
    - Pred order: for each GT-ordered match, the corresponding pred order

    Then computes edit distance between the two order sequences.
    Follows OmniDocBench get_order_paired() methodology.
    """
    # Get matched pairs with both sides present
    paired = [(pi, gi) for pi, gi, _ in matches if pi is not None and gi is not None]
    if len(paired) <= 1:
        return 0.0

    # Sort by GT order to get expected sequence
    paired_sorted = sorted(paired, key=lambda x: gt_elements[x[1]].order)
    gt_order = list(range(len(paired_sorted)))

    # Get pred order: sort by pred order, then map to position
    pred_positions = sorted(range(len(paired_sorted)),
                            key=lambda idx: pred_elements[paired_sorted[idx][0]].order)
    pred_order = [0] * len(paired_sorted)
    for rank, idx in enumerate(pred_positions):
        pred_order[idx] = rank

    # NED on order sequences
    if gt_order == pred_order:
        return 0.0
    return Levenshtein.distance(gt_order, pred_order) / max(len(gt_order), len(pred_order))


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def calculate_structure_eval(prediction: str, reference: str) -> StructureEvalResult:
    """Comprehensive structure evaluation using NED-based metrics.

    Methodology aligned with OmniDocBench (CVPR 2025):
    1. Extract structural elements from both prediction and reference
    2. Match elements using Hungarian algorithm with NED cost
    3. Compute element-level NED (text quality per matched pair)
    4. Compute reading order NED (sequence ordering quality)
    5. Compute element detection F1 (structural element detection)

    Args:
        prediction: Predicted markdown text
        reference: Ground truth markdown text

    Returns:
        StructureEvalResult with NED-based metrics
    """
    pred_elements = extract_elements(prediction)
    gt_elements = extract_elements(reference)

    # Match elements
    matches = _match_elements(pred_elements, gt_elements)

    # Element-level NED (average NED of all matched pairs + unmatched penalties)
    if not matches:
        element_ned = 0.0 if not pred_elements and not gt_elements else 1.0
    else:
        total_ned = sum(ned for _, _, ned in matches)
        element_ned = total_ned / len(matches)

    # Reading Order NED
    reading_order_ned = _reading_order_ned(matches, pred_elements, gt_elements)

    # Element Detection F1
    n_matched = sum(1 for pi, gi, _ in matches if pi is not None and gi is not None)
    n_pred = len(pred_elements)
    n_gt = len(gt_elements)

    precision = n_matched / n_pred if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    recall = n_matched / n_gt if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-type NED breakdown
    type_neds: dict[str, list[float]] = {}
    for pi, gi, ned in matches:
        if pi is not None and gi is not None:
            etype = gt_elements[gi].element_type
            type_neds.setdefault(etype, []).append(ned)
    per_type_ned = {k: sum(v) / len(v) for k, v in type_neds.items()}

    return StructureEvalResult(
        element_ned=element_ned,
        reading_order_ned=reading_order_ned,
        detection_f1=f1,
        detection_precision=precision,
        detection_recall=recall,
        per_type_ned=per_type_ned,
        pred_element_count=n_pred,
        gt_element_count=n_gt,
        matched_count=n_matched,
    )


# ---------------------------------------------------------------------------
# Legacy wrapper
# ---------------------------------------------------------------------------

def calculate_structure_f1(prediction: str, reference: str) -> StructureEvalResult:
    """Legacy wrapper — delegates to calculate_structure_eval().

    Maintains backward compatibility. The .detection_f1 field replaces
    the old element-count-based F1 with proper Hungarian-matched F1.
    The .f1 property provides the same interface as the old StructureF1Result.
    """
    result = calculate_structure_eval(prediction, reference)
    return result

