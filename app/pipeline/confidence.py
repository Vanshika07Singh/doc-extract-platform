"""Confidence scoring.

The model gives us values but no reliable calibrated probability, so we derive a
confidence per field from signals we *can* measure:

  * grounding   - does the value actually appear in the OCR text? (fuzzy match)
  * corroboration - did regex/spaCy independently find the same value?
  * ocr quality - mean OCR word confidence for the document

These are blended into a 0-1 score and compared to a review threshold.
"""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        import json

        return json.dumps(value, ensure_ascii=False)
    return str(value)


def grounding_score(value: Any, source_text: str) -> float:
    """How strongly the value is supported by the source text (0-1)."""
    text = _stringify(value).strip()
    if not text:
        return 0.0
    if not source_text:
        return 0.0
    # partial_ratio handles substrings / minor OCR noise well.
    score = fuzz.partial_ratio(text.lower(), source_text.lower()) / 100.0
    return round(score, 4)


def field_confidence(
    value: Any,
    source_text: str,
    ocr_confidence: float,
    corroborated: bool,
) -> float:
    """Blend signals into a single confidence value."""
    if value is None or value == [] or value == "":
        return 0.0

    grounding = grounding_score(value, source_text)
    base = 0.45
    score = base + 0.35 * grounding + (0.20 if corroborated else 0.0)

    # Scale by OCR quality so noisy scans are penalized but never zeroed out.
    ocr_factor = 0.7 + 0.3 * max(0.0, min(ocr_confidence, 1.0))
    score *= ocr_factor

    return round(max(0.0, min(score, 1.0)), 4)
