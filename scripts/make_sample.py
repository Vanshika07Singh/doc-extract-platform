"""Generate a sample invoice PNG so you can test the OCR pipeline end-to-end.

Usage:
    python scripts/make_sample.py            # writes samples/invoice.png
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LINES = [
    "ACME CORPORATION",
    "",
    "INVOICE",
    "Invoice Number: INV-2024-0042",
    "Invoice Date: 2024-03-15",
    "Due Date: 2024-04-14",
    "",
    "Bill To:",
    "Globex Inc",
    "500 Market Street, San Francisco, CA 94105",
    "",
    "Description           Qty    Unit Price    Amount",
    "Consulting services    10        100.00    1000.00",
    "Support plan            1        200.00     200.00",
    "",
    "Subtotal:   1200.00",
    "Tax (8%):     96.00",
    "Total Due:  1296.00 USD",
]


def _load_font(size: int):
    """Prefer a clear monospace font so OCR reads decimals/columns reliably."""
    for name in ("Menlo.ttc", "DejaVuSansMono.ttf", "Courier New.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "samples"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "invoice.png"

    font = _load_font(28)
    line_h = 46
    width = 1100
    height = line_h * len(LINES) + 80
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    y = 40
    for line in LINES:
        draw.text((50, y), line, fill="black", font=font)
        y += line_h

    img.save(out_path, dpi=(300, 300))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
