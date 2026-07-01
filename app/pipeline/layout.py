"""Lightweight layout analysis.

A full LayoutParser/Detectron2 stack is heavy and awkward to install on most
machines, so this module reconstructs document structure geometrically from the
word boxes produced by the OCR stage:

  * groups words into lines (by vertical proximity within a page)
  * groups lines into blocks (by vertical gaps)
  * exposes a clean reading-order text plus a rough header / body split

The block structure is fed to the extractor as extra context, which improves
LLM extraction on multi-column or table-heavy documents.

To swap in true LayoutParser later, replace ``analyze_layout`` with a version
that returns the same ``LayoutResult`` shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.pipeline.ocr import Word


@dataclass
class Block:
    text: str
    page: int
    top: int
    bottom: int
    confidence: float


@dataclass
class LayoutResult:
    reading_order_text: str
    blocks: list[Block] = field(default_factory=list)
    header_text: str = ""

    def as_context(self) -> str:
        """Compact, structure-aware text block handed to the extractor."""
        if not self.blocks:
            return self.reading_order_text
        return "\n\n".join(b.text for b in self.blocks)


def _group_lines(words: list[Word], y_tol: int = 8) -> list[list[Word]]:
    lines: list[list[Word]] = []
    for w in sorted(words, key=lambda x: (x.page, x.top, x.left)):
        placed = False
        for line in reversed(lines):
            ref = line[-1]
            if ref.page == w.page and abs(ref.top - w.top) <= y_tol:
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    return lines


def analyze_layout(words: list[Word], fallback_text: str = "") -> LayoutResult:
    if not words:
        return LayoutResult(reading_order_text=fallback_text)

    lines = _group_lines(words)
    line_records = []
    for line in lines:
        ordered = sorted(line, key=lambda x: x.left)
        text = " ".join(w.text for w in ordered)
        top = min(w.top for w in ordered)
        bottom = max(w.top + w.height for w in ordered)
        conf = sum(w.confidence for w in ordered) / len(ordered)
        page = ordered[0].page
        line_records.append((page, top, bottom, text, conf))

    line_records.sort(key=lambda r: (r[0], r[1]))

    # Group lines into blocks using the median line height as the gap threshold.
    heights = [b - t for _, t, b, _, _ in line_records if b > t]
    median_h = sorted(heights)[len(heights) // 2] if heights else 12
    gap_threshold = median_h * 1.8

    blocks: list[Block] = []
    cur: list[tuple] = []
    for rec in line_records:
        if not cur:
            cur = [rec]
            continue
        prev = cur[-1]
        same_page = rec[0] == prev[0]
        gap = rec[1] - prev[2]
        if same_page and gap <= gap_threshold:
            cur.append(rec)
        else:
            blocks.append(_finalize_block(cur))
            cur = [rec]
    if cur:
        blocks.append(_finalize_block(cur))

    reading_order_text = "\n".join(r[3] for r in line_records)
    header_text = blocks[0].text if blocks else ""
    return LayoutResult(
        reading_order_text=reading_order_text,
        blocks=blocks,
        header_text=header_text,
    )


def _finalize_block(lines: list[tuple]) -> Block:
    text = "\n".join(r[3] for r in lines)
    page = lines[0][0]
    top = min(r[1] for r in lines)
    bottom = max(r[2] for r in lines)
    conf = sum(r[4] for r in lines) / len(lines)
    return Block(text=text, page=page, top=top, bottom=bottom, confidence=conf)
