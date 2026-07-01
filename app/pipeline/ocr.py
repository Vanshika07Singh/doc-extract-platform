"""OCR layer.

Strategy:
  * PDFs that contain a real text layer are read directly with pdfplumber
    (fast, lossless, no OCR needed).
  * Scanned PDFs and images fall back to Tesseract OCR.

Returns text plus word-level bounding boxes and per-word confidence so the
layout and confidence-scoring stages have something to work with.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image


@dataclass
class Word:
    text: str
    confidence: float  # 0-1
    left: int
    top: int
    width: int
    height: int
    page: int


@dataclass
class OcrResult:
    text: str
    words: list[Word] = field(default_factory=list)
    pages: int = 0
    engine: str = "tesseract"

    @property
    def mean_word_confidence(self) -> float:
        confs = [w.confidence for w in self.words if w.confidence >= 0]
        return round(sum(confs) / len(confs), 4) if confs else 0.0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _image_to_words(image: Image.Image, page: int, lang: str) -> tuple[str, list[Word]]:
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    words: list[Word] = []
    lines: list[str] = []
    current_line_num: Optional[int] = None
    current_line: list[str] = []

    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        conf = float(data["conf"][i])
        line_id = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if line_id != current_line_num:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = []
            current_line_num = line_id
        if not text:
            continue
        current_line.append(text)
        words.append(
            Word(
                text=text,
                confidence=max(conf, 0.0) / 100.0,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
                page=page,
            )
        )
    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(lines), words


def _ocr_image_bytes(content: bytes, lang: str) -> OcrResult:
    image = Image.open(io.BytesIO(content)).convert("RGB")
    text, words = _image_to_words(image, page=1, lang=lang)
    return OcrResult(text=text, words=words, pages=1, engine="tesseract")


def _read_pdf(content: bytes, lang: str) -> OcrResult:
    import pdfplumber

    text_chunks: list[str] = []
    words: list[Word] = []
    needs_ocr_pages: list[int] = []

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        page_count = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)
                for w in page.extract_words():
                    words.append(
                        Word(
                            text=w["text"],
                            confidence=0.99,  # native text layer = high confidence
                            left=int(w["x0"]),
                            top=int(w["top"]),
                            width=int(w["x1"] - w["x0"]),
                            height=int(w["bottom"] - w["top"]),
                            page=idx,
                        )
                    )
            else:
                needs_ocr_pages.append(idx)

    if needs_ocr_pages:
        # Scanned pages: rasterize and OCR them.
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(content, dpi=300)
        for idx in needs_ocr_pages:
            if idx - 1 < len(images):
                ptext, pwords = _image_to_words(images[idx - 1], page=idx, lang=lang)
                text_chunks.append(ptext)
                words.extend(pwords)

    engine = "pdfplumber" if not needs_ocr_pages else "pdfplumber+tesseract"
    return OcrResult(
        text="\n\n".join(text_chunks).strip(),
        words=words,
        pages=page_count,
        engine=engine,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def run_ocr(content: bytes, filename: str, lang: str = "eng") -> OcrResult:
    """Extract text + word boxes from a PDF or image file."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _read_pdf(content, lang)
    return _ocr_image_bytes(content, lang)
