"""Normalized Edit Distance (NED) metric.

Reference: OmniDocBench (Ouyang et al., CVPR 2025)
Formula: NED = edit_distance(pred, gt) / max(len(pred), len(gt))

Lower is better. 0.0 = perfect match, 1.0 = completely different.
"""

import Levenshtein


def calculate_ned(prediction: str, reference: str) -> float:
    """Calculate Normalized Edit Distance.

    Args:
        prediction: Predicted text.
        reference: Ground truth text.

    Returns:
        NED score in [0, 1]. 0.0 = perfect match.
    """
    if not prediction and not reference:
        return 0.0
    upper = max(len(prediction), len(reference))
    if upper == 0:
        return 0.0
    return Levenshtein.distance(prediction, reference) / upper


def calculate_ned_batch(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Calculate NED for a batch of samples.

    Returns dict with:
    - edit_whole: total_edits / total_max_lengths (corpus-level)
    - sample_avg: mean of per-sample NED
    - page_avg: mean of per-page aggregated NED
    """
    if not predictions or not references:
        return {"edit_whole": 0.0, "sample_avg": 0.0}

    total_edits = 0
    total_upper = 0
    sample_neds = []

    for pred, ref in zip(predictions, references):
        upper = max(len(pred), len(ref))
        if upper == 0:
            continue
        edit = Levenshtein.distance(pred, ref)
        total_edits += edit
        total_upper += upper
        sample_neds.append(edit / upper)

    edit_whole = total_edits / total_upper if total_upper > 0 else 0.0
    sample_avg = sum(sample_neds) / len(sample_neds) if sample_neds else 0.0

    return {
        "edit_whole": edit_whole,
        "sample_avg": sample_avg,
    }
