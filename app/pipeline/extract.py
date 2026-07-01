"""Entity extraction coordinator.

Combines the LLM extraction with deterministic baselines (regex + spaCy),
validates the result against the typed Pydantic schema, and attaches a
confidence score to every field.
"""

from __future__ import annotations

import re
from typing import Any

from app.config import Settings
from app.pipeline import regex_extract, spacy_ner
from app.pipeline.confidence import field_confidence
from app.pipeline.llm_extract import llm_extract
from app.schemas.documents import DocumentType, schema_for
from app.schemas.response import FieldResult


_ORG_STOPWORDS = re.compile(
    r"\b(invoice|receipt|date|number|no\.?|total|subtotal|tax|bill\s*to|ship\s*to"
    r"|description|qty|quantity|unit\s*price|amount|price|due)\b",
    re.IGNORECASE,
)


def _first_meaningful_line(text: str) -> str | None:
    """Return the first non-empty line that isn't an obvious header/keyword."""
    for line in text.splitlines():
        s = line.strip(" \t'\"`")
        if len(s) > 2 and not _ORG_STOPWORDS.search(s):
            return s
    return None


def _looks_like_org(s: str) -> bool:
    if _ORG_STOPWORDS.search(s):
        return False
    if any(ch.isdigit() for ch in s):
        return False
    return len(s.split()) <= 5


def _guess_org(text: str, ner: dict[str, list[str]]) -> str | None:
    """Pick a likely organisation name, avoiding label/table artefacts."""
    for org in ner.get("ORG", []):
        if _looks_like_org(org):
            return org
    return _first_meaningful_line(text)


def _baseline_extract(doc_type: DocumentType, text: str) -> dict[str, Any]:
    """Regex + spaCy fallback used when the LLM is disabled or as corroboration.

    This is intentionally rich so the platform produces useful structured output
    even with no API key (offline mode) or when the LLM call fails.
    """
    ner = spacy_ner.extract_entities(text)
    emails = regex_extract.find_emails(text)
    phones = regex_extract.find_phones(text)
    dates = regex_extract.find_dates(text)
    first_date = dates[0] if dates else None

    data: dict[str, Any] = {}

    if doc_type in (DocumentType.INVOICE, DocumentType.RECEIPT, DocumentType.PURCHASE_ORDER):
        data["currency"] = regex_extract.find_currency(text)
        data["subtotal"] = regex_extract.labeled_amount(text, r"sub\s*total")
        data["tax"] = regex_extract.labeled_amount(text, r"tax|vat|gst")
        data["total"] = regex_extract.labeled_amount(
            text, r"total\s*due|amount\s*due|grand\s*total|balance\s*due|\btotal\b"
        )
        data["items"] = regex_extract.parse_line_items(text)

    if doc_type == DocumentType.INVOICE:
        data["invoice_number"] = regex_extract.find_reference_number(
            text, r"invoice\s*(?:no\.?|number|#)"
        )
        data["invoice_date"] = regex_extract.labeled_date(text, r"invoice\s*date") or first_date
        data["due_date"] = regex_extract.labeled_date(text, r"due\s*date")
        data["vendor_name"] = _guess_org(text, ner)
        data["billing_address"] = regex_extract.labeled_block(text, r"bill\s*to")
        data["shipping_address"] = regex_extract.labeled_block(text, r"ship\s*to")
    elif doc_type == DocumentType.RECEIPT:
        data["merchant_name"] = _guess_org(text, ner)
        data["transaction_date"] = first_date
        data["payment_method"] = regex_extract.labeled_value(text, r"payment\s*method|paid\s*(?:by|with)")
    elif doc_type == DocumentType.PURCHASE_ORDER:
        data["po_number"] = regex_extract.find_reference_number(
            text, r"p\.?\s*o\.?\s*(?:no\.?|number|#)|purchase\s*order\s*(?:no\.?|number|#)?"
        )
        data["order_date"] = regex_extract.labeled_date(text, r"order\s*date") or first_date
        data["supplier_name"] = _guess_org(text, ner)
        data["buyer_name"] = regex_extract.labeled_value(text, r"buyer|bill\s*to")
        data["shipping_address"] = regex_extract.labeled_block(text, r"ship\s*to")
    elif doc_type == DocumentType.RESUME:
        data["full_name"] = (ner.get("PERSON") or [None])[0] or _first_meaningful_line(text)
        data["email"] = emails[0] if emails else None
        data["phone"] = phones[0] if phones else None
        data["location"] = (ner.get("GPE") or [None])[0]
        skills = regex_extract.labeled_value(text, r"skills|technical\s*skills")
        if skills:
            data["skills"] = [s.strip() for s in re.split(r"[,;|]", skills) if s.strip()]
    elif doc_type == DocumentType.CONTRACT:
        data["title"] = _first_meaningful_line(text)
        data["parties"] = ner.get("ORG", [])[:4]
        data["effective_date"] = regex_extract.labeled_date(text, r"effective\s*date") or first_date
        data["expiration_date"] = regex_extract.labeled_date(text, r"expir|termination")
        data["governing_law"] = regex_extract.labeled_value(text, r"governing\s*law")
    elif doc_type == DocumentType.ID_DOCUMENT:
        data["full_name"] = (ner.get("PERSON") or [None])[0]
        data["date_of_birth"] = regex_extract.labeled_date(text, r"date\s*of\s*birth|dob") or first_date
        data["document_number"] = regex_extract.find_reference_number(
            text, r"(?:passport|licen[sc]e|document|id|card)\s*(?:no\.?|number|#)"
        )
        data["nationality"] = regex_extract.labeled_value(text, r"nationality")
        data["expiry_date"] = regex_extract.labeled_date(text, r"date\s*of\s*expiry|expir")

    return {k: v for k, v in data.items() if v not in (None, "", [], {})}


def _corroboration_set(text: str) -> set[str]:
    """Lowercased values found by deterministic extractors, for matching."""
    found: set[str] = set()
    for fn in (regex_extract.find_emails, regex_extract.find_phones,
               regex_extract.find_money, regex_extract.find_dates):
        for v in fn(text):
            found.add(v.lower().strip())
    for vals in spacy_ner.extract_entities(text).values():
        for v in vals:
            found.add(v.lower().strip())
    return found


def _is_corroborated(value: Any, corroboration: set[str]) -> bool:
    if value is None:
        return False
    s = str(value).lower().strip()
    if not s:
        return False
    return any(s in c or c in s for c in corroboration)


def extract_fields(
    doc_type: DocumentType,
    text: str,
    layout_context: str,
    ocr_confidence: float,
    settings: Settings,
) -> tuple[dict[str, Any], dict[str, FieldResult], list[str]]:
    """Run extraction and return (validated_data, field_results, warnings)."""
    warnings: list[str] = []
    model_cls = schema_for(doc_type)

    raw: dict[str, Any] = {}
    sources: dict[str, str] = {}

    if settings.llm_enabled:
        llm_out = llm_extract(
            doc_type=doc_type,
            text=text,
            layout_context=layout_context,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        )
        if llm_out and "__error__" in llm_out:
            warnings.append(f"LLM extraction failed, using baseline: {llm_out['__error__']}")
            raw = _baseline_extract(doc_type, text)
            sources = {k: "spacy" for k in raw}
        elif llm_out:
            raw = llm_out
            sources = {k: "llm" for k in raw}
    else:
        warnings.append("LLM disabled (no OPENAI_API_KEY). Using regex + spaCy baseline.")
        raw = _baseline_extract(doc_type, text)
        sources = {k: "spacy" for k in raw}
        if not spacy_ner.is_available():
            warnings.append(
                "spaCy model 'en_core_web_sm' not installed; only regex extraction is active."
            )

    # Validate / coerce against the typed schema. Drop unknown keys.
    allowed = set(model_cls.model_fields.keys())
    filtered = {k: v for k, v in raw.items() if k in allowed}
    try:
        validated = model_cls.model_validate(filtered)
        data = validated.model_dump()
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Schema validation issue: {exc}")
        data = {k: filtered.get(k) for k in allowed}

    # Confidence per field.
    corroboration = _corroboration_set(text)
    grounding_source = text or layout_context
    fields: dict[str, FieldResult] = {}
    for key, value in data.items():
        corroborated = _is_corroborated(value, corroboration)
        conf = field_confidence(value, grounding_source, ocr_confidence, corroborated)
        source = "fusion" if (sources.get(key) == "llm" and corroborated) else sources.get(key, "llm")
        fields[key] = FieldResult(
            value=value,
            confidence=conf,
            source=source,
            needs_review=conf < settings.review_threshold and _has_value(value),
        )

    return data, fields, warnings


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})
