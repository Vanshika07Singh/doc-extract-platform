"""End-to-end pipeline orchestration.

    bytes -> OCR -> layout -> classify -> extract -> validate -> scored JSON
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.pipeline import classify, extract, layout, ocr
from app.schemas.documents import DocumentType
from app.schemas.response import ExtractionResponse, OcrMeta


def process_document(
    content: bytes,
    filename: str,
    settings: Settings | None = None,
    forced_type: DocumentType | None = None,
    include_raw_text: bool = True,
) -> ExtractionResponse:
    settings = settings or get_settings()

    # 1. OCR
    ocr_result = ocr.run_ocr(content, filename, lang=settings.ocr_lang)

    # 2. Layout analysis
    layout_result = layout.analyze_layout(ocr_result.words, fallback_text=ocr_result.text)
    layout_context = layout_result.as_context()

    # 3. Classification
    if forced_type and forced_type != DocumentType.UNKNOWN:
        doc_type, type_conf = forced_type, 1.0
    else:
        doc_type, type_conf = classify.classify(ocr_result.text or layout_context)
        if doc_type == DocumentType.UNKNOWN:
            doc_type = DocumentType.INVOICE  # safe default schema

    # 4 + 5. Extraction + validation + confidence
    data, fields, warnings = extract.extract_fields(
        doc_type=doc_type,
        text=ocr_result.text,
        layout_context=layout_context,
        ocr_confidence=ocr_result.mean_word_confidence,
        settings=settings,
    )

    scored = [f.confidence for f in fields.values() if f.value not in (None, "", [], {})]
    overall = round(sum(scored) / len(scored), 4) if scored else 0.0
    needs_review = [name for name, f in fields.items() if f.needs_review]

    if not ocr_result.text.strip():
        warnings.append("OCR produced no text; check that the file is a readable document.")

    return ExtractionResponse(
        document_type=doc_type,
        type_confidence=type_conf,
        data=data,
        fields=fields,
        overall_confidence=overall,
        needs_review=needs_review,
        warnings=warnings,
        ocr=OcrMeta(
            engine=ocr_result.engine,
            pages=ocr_result.pages,
            mean_word_confidence=ocr_result.mean_word_confidence,
            char_count=len(ocr_result.text),
        ),
        raw_text=ocr_result.text if include_raw_text else None,
    )
