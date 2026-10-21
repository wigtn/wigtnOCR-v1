"""Element matching for OmniDocBench evaluation.

Matches GT elements to prediction elements using Hungarian algorithm
with Normalized Edit Distance cost matrix.

Based on OmniDocBench's match_quick.py.
Reference: OmniDocBench (Ouyang et al., CVPR 2025)
"""

import Levenshtein
import numpy as np
from scipy.optimize import linear_sum_assignment

from evaluation.omnidocbench.normalizer import clean_string, normalize_formula, textblock2unicode


def match_text_elements(
    gt_elements: list,
    pred_elements: list,
    normalize: bool = True,
) -> list[dict]:
    """Match GT text elements to prediction text elements via Hungarian algorithm.

    Args:
        gt_elements: List of GTElement (text blocks).
        pred_elements: List of PredElement (text blocks).
        normalize: Apply clean_string normalization before matching.

    Returns:
        List of match dicts with keys:
            gt_idx, gt, pred_idx, pred, norm_gt, norm_pred,
            gt_category_type, gt_attribute, img_id, edit, matched
    """
    if not gt_elements and not pred_elements:
        return []

    # Extract and normalize text
    gt_texts = []
    for el in gt_elements:
        text = el.content if hasattr(el, "content") else el.get("content", "")
        gt_texts.append(text)

    pred_texts = []
    for el in pred_elements:
        text = el.content if hasattr(el, "content") else el.get("content", "")
        pred_texts.append(text)

    if normalize:
        gt_norm = [clean_string(textblock2unicode(t)) for t in gt_texts]
        pred_norm = [clean_string(textblock2unicode(t)) for t in pred_texts]
    else:
        gt_norm = gt_texts
        pred_norm = pred_texts

    # Compute NED cost matrix
    n_gt = len(gt_norm)
    n_pred = len(pred_norm)

    if n_gt == 0:
        # All predictions are unmatched
        return [
            _make_match(None, i, "", pred_texts[i], "", pred_norm[i], 1.0, pred_el=pred_elements[i])
            for i in range(n_pred)
        ]
    if n_pred == 0:
        # All GT are unmatched
        return [
            _make_match(i, None, gt_texts[i], "", gt_norm[i], "", 1.0, gt_el=gt_elements[i])
            for i in range(n_gt)
        ]

    cost = np.ones((n_gt, n_pred), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_pred):
            upper = max(len(gt_norm[i]), len(pred_norm[j]))
            if upper > 0:
                cost[i, j] = Levenshtein.distance(gt_norm[i], pred_norm[j]) / upper
            else:
                cost[i, j] = 0.0

    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_gt = set()
    matched_pred = set()
    results = []

    for i, j in zip(row_ind, col_ind):
        results.append(_make_match(
            i, j, gt_texts[i], pred_texts[j],
            gt_norm[i], pred_norm[j], cost[i, j],
            gt_el=gt_elements[i], pred_el=pred_elements[j],
        ))
        matched_gt.add(i)
        matched_pred.add(j)

    # Unmatched GT
    for i in range(n_gt):
        if i not in matched_gt:
            results.append(_make_match(
                i, None, gt_texts[i], "", gt_norm[i], "", 1.0,
                gt_el=gt_elements[i],
            ))

    # Unmatched predictions
    for j in range(n_pred):
        if j not in matched_pred:
            results.append(_make_match(
                None, j, "", pred_texts[j], "", pred_norm[j], 1.0,
                pred_el=pred_elements[j],
            ))

    return results


def match_formula_elements(
    gt_elements: list,
    pred_elements: list,
) -> list[dict]:
    """Match GT formulas to prediction formulas.

    Uses normalize_formula() for comparison.
    """
    gt_texts = []
    for el in gt_elements:
        latex = el.latex if hasattr(el, "latex") else el.get("latex", "")
        text = el.content if hasattr(el, "content") else el.get("content", "")
        gt_texts.append(latex or text)

    pred_texts = [
        el.content if hasattr(el, "content") else el.get("content", "")
        for el in pred_elements
    ]

    gt_norm = [normalize_formula(t) for t in gt_texts]
    pred_norm = [normalize_formula(t) for t in pred_texts]

    n_gt, n_pred = len(gt_norm), len(pred_norm)
    if n_gt == 0 or n_pred == 0:
        results = []
        for i in range(n_gt):
            results.append(_make_match(i, None, gt_texts[i], "", gt_norm[i], "", 1.0, gt_el=gt_elements[i]))
        for j in range(n_pred):
            results.append(_make_match(None, j, "", pred_texts[j], "", pred_norm[j], 1.0, pred_el=pred_elements[j]))
        return results

    cost = np.ones((n_gt, n_pred), dtype=np.float64)
    for i in range(n_gt):
        for j in range(n_pred):
            upper = max(len(gt_norm[i]), len(pred_norm[j]))
            if upper > 0:
                cost[i, j] = Levenshtein.distance(gt_norm[i], pred_norm[j]) / upper
            else:
                cost[i, j] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_gt, matched_pred = set(), set()
    results = []

    for i, j in zip(row_ind, col_ind):
        results.append(_make_match(
            i, j, gt_texts[i], pred_texts[j],
            gt_norm[i], pred_norm[j], cost[i, j],
            gt_el=gt_elements[i], pred_el=pred_elements[j],
        ))
        matched_gt.add(i)
        matched_pred.add(j)

    for i in range(n_gt):
        if i not in matched_gt:
            results.append(_make_match(i, None, gt_texts[i], "", gt_norm[i], "", 1.0, gt_el=gt_elements[i]))
    for j in range(n_pred):
        if j not in matched_pred:
            results.append(_make_match(None, j, "", pred_texts[j], "", pred_norm[j], 1.0, pred_el=pred_elements[j]))

    return results


def compute_reading_order_ned(
    text_matches: list[dict],
    gt_elements: list,
) -> float:
    """Compute reading order NED from matched text elements.

    Compares the GT reading order vs the order in which matched predictions appear.

    Reference: OmniDocBench get_order_paired() method.

    Returns:
        NED score for reading order. 0.0 = perfect order, 1.0 = completely wrong.
    """
    # Get matched pairs with both sides
    paired = []
    for m in text_matches:
        if m.get("gt_idx") is not None and m.get("pred_idx") is not None:
            gt_idx = m["gt_idx"]
            pred_idx = m["pred_idx"]
            # GT order from element's order attribute
            gt_order = gt_elements[gt_idx].order if hasattr(gt_elements[gt_idx], "order") else gt_idx
            if gt_order is None:
                gt_order = gt_idx
            paired.append((gt_order, pred_idx))

    if len(paired) <= 1:
        return 0.0

    # Sort by GT order → gives expected sequence
    paired.sort(key=lambda x: x[0])
    gt_sequence = list(range(len(paired)))

    # Sort by pred order → gives actual sequence
    pred_order_sorted = sorted(range(len(paired)), key=lambda idx: paired[idx][1])
    pred_sequence = [0] * len(paired)
    for rank, idx in enumerate(pred_order_sorted):
        pred_sequence[idx] = rank

    if gt_sequence == pred_sequence:
        return 0.0

    return Levenshtein.distance(gt_sequence, pred_sequence) / max(len(gt_sequence), len(pred_sequence))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_match(
    gt_idx, pred_idx, gt_text, pred_text,
    norm_gt, norm_pred, edit_dist,
    gt_el=None, pred_el=None,
) -> dict:
    """Create a match result dict."""
    result = {
        "gt_idx": gt_idx,
        "pred_idx": pred_idx,
        "gt": gt_text,
        "pred": pred_text,
        "norm_gt": norm_gt,
        "norm_pred": norm_pred,
        "edit": edit_dist,
        "matched": gt_idx is not None and pred_idx is not None,
    }
    if gt_el:
        cat = gt_el.category if hasattr(gt_el, "category") else gt_el.get("category", "")
        attrs = gt_el.attributes if hasattr(gt_el, "attributes") else gt_el.get("attributes", {})
        result["gt_category_type"] = cat
        result["gt_attribute"] = attrs
    if pred_el:
        cat = pred_el.category if hasattr(pred_el, "category") else pred_el.get("category", "")
        result["pred_category_type"] = cat
    return result
