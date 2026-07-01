"""spaCy NER baseline.

Loads ``en_core_web_sm`` lazily and caches it. If the model is not installed,
the loader returns ``None`` and the rest of the pipeline degrades gracefully to
regex-only corroboration.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional


@lru_cache(maxsize=1)
def _load_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def extract_entities(text: str, max_chars: int = 20000) -> dict[str, list[str]]:
    """Return entities grouped by spaCy label (PERSON, ORG, GPE, DATE, MONEY...)."""
    nlp = _load_nlp()
    if nlp is None or not text:
        return {}

    doc = nlp(text[:max_chars])
    grouped: dict[str, list[str]] = {}
    for ent in doc.ents:
        grouped.setdefault(ent.label_, [])
        value = ent.text.strip()
        if value and value not in grouped[ent.label_]:
            grouped[ent.label_].append(value)
    return grouped


def is_available() -> bool:
    return _load_nlp() is not None
