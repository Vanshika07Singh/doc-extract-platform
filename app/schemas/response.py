"""API response envelope: validated data + per-field confidence scoring."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.schemas.documents import DocumentType


class FieldResult(BaseModel):
    """A single extracted field with its confidence and provenance."""

    value: Any = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: str = Field(
        default="llm", description="Where the value came from: llm | spacy | regex | fusion"
    )
    needs_review: bool = False


class OcrMeta(BaseModel):
    engine: str = "tesseract"
    pages: int = 0
    mean_word_confidence: float = 0.0
    char_count: int = 0


class ExtractionResponse(BaseModel):
    document_type: DocumentType
    type_confidence: float = Field(ge=0.0, le=1.0, default=0.0)

    # Clean, flat structured output (validated against the type's schema).
    data: dict[str, Any] = Field(default_factory=dict)

    # Per-field confidence + provenance, keyed by field name.
    fields: dict[str, FieldResult] = Field(default_factory=dict)

    # Aggregate signals for downstream routing / human-in-the-loop.
    overall_confidence: float = 0.0
    needs_review: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    ocr: OcrMeta = Field(default_factory=OcrMeta)
    raw_text: Optional[str] = None
