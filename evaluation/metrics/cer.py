"""Character Error Rate (CER) and Word Error Rate (WER) metrics.

Extracted from src/eval_parsers.py and cleaned into simple functional interfaces.
Uses the jiwer library for edit-distance-based error rate computation.
"""

import jiwer


def calculate_cer(prediction: str, reference: str) -> float:
    """Calculate Character Error Rate between prediction and reference.

    Args:
        prediction: Predicted/parsed text (hypothesis)
        reference: Ground truth text

    Returns:
        CER value in [0, inf). 0.0 = perfect match.
    """
    if not reference:
        return 0.0 if not prediction else float("inf")
    if not prediction:
        return 1.0

    return jiwer.cer(reference, prediction)


def calculate_wer(prediction: str, reference: str) -> float:
    """Calculate Word Error Rate between prediction and reference.

    Uses whitespace tokenization by default.

    Args:
        prediction: Predicted/parsed text (hypothesis)
        reference: Ground truth text

    Returns:
        WER value in [0, inf). 0.0 = perfect match.
    """
    if not reference:
        return 0.0 if not prediction else float("inf")
    if not prediction:
        return 1.0

    return jiwer.wer(reference, prediction)


def calculate_edit_distance(prediction: str, reference: str) -> int:
    """Calculate character-level edit distance (Levenshtein distance).

    Args:
        prediction: Predicted text
        reference: Ground truth text

    Returns:
        Integer edit distance (number of insertions, deletions, substitutions)
    """
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)

    output = jiwer.process_characters(reference, prediction)
    return output.substitutions + output.deletions + output.insertions
