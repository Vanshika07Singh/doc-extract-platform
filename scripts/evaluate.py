"""Evaluate extraction quality on a labeled JSONL benchmark.

The included benchmark is intentionally small and synthetic. It exists to make
the evaluation loop concrete; replace/extend it with public or internal labeled
documents before quoting metrics on a resume.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --benchmark data/benchmark.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import Settings
from app.pipeline import classify, extract
from app.schemas.documents import DocumentType


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.lower().strip().split())
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in sorted(value.items())}
    return value


def _matches(predicted: Any, expected: Any) -> bool:
    """Compare values with light normalization for strings and numbers."""
    if isinstance(expected, float) and isinstance(predicted, (int, float)):
        return abs(float(predicted) - expected) < 0.001
    return _normalize(predicted) == _normalize(expected)


def load_examples(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def evaluate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    settings = Settings(openai_api_key="")  # deterministic offline baseline

    total_expected = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    classified_correctly = 0
    rows: list[dict[str, Any]] = []

    for example in examples:
        doc_type = DocumentType(example["document_type"])
        text = example["text"]
        expected = example["expected"]

        predicted_type, _ = classify.classify(text)
        if predicted_type == doc_type:
            classified_correctly += 1

        data, _, _ = extract.extract_fields(
            doc_type=doc_type,
            text=text,
            layout_context=text,
            ocr_confidence=0.95,
            settings=settings,
        )

        matched_fields = 0
        total_expected += len(expected)
        for field, expected_value in expected.items():
            predicted_value = data.get(field)
            if _matches(predicted_value, expected_value):
                true_positive += 1
                matched_fields += 1
            else:
                false_negative += 1

        for field, predicted_value in data.items():
            if field not in expected and _has_value(predicted_value):
                false_positive += 1

        rows.append(
            {
                "id": example["id"],
                "document_type": doc_type.value,
                "matched": matched_fields,
                "expected": len(expected),
            }
        )

    precision_den = true_positive + false_positive
    recall_den = true_positive + false_negative
    precision = true_positive / precision_den if precision_den else 0.0
    recall = true_positive / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    field_accuracy = true_positive / total_expected if total_expected else 0.0
    classification_accuracy = classified_correctly / len(examples) if examples else 0.0

    return {
        "documents": len(examples),
        "classification_accuracy": classification_accuracy,
        "field_accuracy": field_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate field extraction quality.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/benchmark.jsonl"),
        help="Path to JSONL benchmark file.",
    )
    args = parser.parse_args()

    report = evaluate(load_examples(args.benchmark))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
