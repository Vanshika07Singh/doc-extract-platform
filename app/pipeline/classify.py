"""Document-type classification.

Heuristic, offline-first classifier based on weighted keyword signals. It is
fast, explainable, and good enough to route to the right schema. The LLM stage
can still correct the structured output, but choosing the schema up front keeps
extraction focused.
"""

from __future__ import annotations

import re

from app.schemas.documents import DocumentType

# Each signal: (regex, weight). Higher weight = stronger evidence.
_SIGNALS: dict[DocumentType, list[tuple[str, float]]] = {
    DocumentType.INVOICE: [
        (r"\binvoice\b", 3.0),
        (r"\binvoice\s*(no|number|#)\b", 4.0),
        (r"\bbill\s*to\b", 2.0),
        (r"\bsubtotal\b", 1.5),
        (r"\bamount\s*due\b", 2.0),
    ],
    DocumentType.RECEIPT: [
        (r"\breceipt\b", 3.0),
        (r"\bchange\b", 1.0),
        (r"\bcash\b|\bvisa\b|\bmastercard\b", 1.5),
        (r"\bthank you\b", 1.0),
        (r"\bmerchant\b", 1.5),
    ],
    DocumentType.PURCHASE_ORDER: [
        (r"\bpurchase\s*order\b", 4.0),
        (r"\bp\.?o\.?\s*(no|number|#)\b", 3.5),
        (r"\bsupplier\b", 1.5),
        (r"\bship\s*to\b", 1.0),
    ],
    DocumentType.RESUME: [
        (r"\bcurriculum vitae\b|\bresume\b", 4.0),
        (r"\bwork experience\b|\bprofessional experience\b", 3.0),
        (r"\beducation\b", 1.5),
        (r"\bskills\b", 1.5),
        (r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", 1.0),
    ],
    DocumentType.CONTRACT: [
        (r"\bagreement\b", 3.0),
        (r"\bthis (agreement|contract)\b", 3.5),
        (r"\bparties\b|\bparty\b", 1.5),
        (r"\bgoverning law\b", 2.5),
        (r"\bwhereas\b", 2.0),
        (r"\bhereinafter\b", 2.0),
    ],
    DocumentType.ID_DOCUMENT: [
        (r"\bpassport\b", 4.0),
        (r"\bdriver'?s? licen[sc]e\b", 4.0),
        (r"\bnational id\b|\bidentity card\b", 3.5),
        (r"\bdate of birth\b|\bdob\b", 2.0),
        (r"\bnationality\b", 1.5),
    ],
}


def classify(text: str) -> tuple[DocumentType, float]:
    """Return (document_type, confidence in 0-1)."""
    if not text or not text.strip():
        return DocumentType.UNKNOWN, 0.0

    lowered = text.lower()
    scores: dict[DocumentType, float] = {}
    for doc_type, signals in _SIGNALS.items():
        score = 0.0
        for pattern, weight in signals:
            if re.search(pattern, lowered):
                score += weight
        scores[doc_type] = score

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    if best_score == 0:
        return DocumentType.UNKNOWN, 0.0

    total = sum(scores.values()) or 1.0
    # Confidence blends share-of-evidence with absolute strength.
    share = best_score / total
    strength = min(best_score / 6.0, 1.0)
    confidence = round(0.5 * share + 0.5 * strength, 4)
    return best_type, confidence
