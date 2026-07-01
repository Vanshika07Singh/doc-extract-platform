"""Lightweight pipeline tests that don't require OCR binaries or an API key.

These exercise the classifier, schema validation, and confidence scoring on
plain text so they run fast and offline.
"""

from app.config import Settings
from app.pipeline import classify, extract, regex_extract
from app.pipeline.confidence import field_confidence, grounding_score
from app.schemas.documents import DocumentType

INVOICE_TEXT = """
ACME Corporation
INVOICE
Invoice Number: INV-2024-0042
Invoice Date: 2024-03-15
Bill To: Globex Inc, 500 Market St, San Francisco, CA
Subtotal: 1200.00
Tax: 96.00
Amount Due: 1296.00 USD
"""

INVOICE_TABLE_TEXT = """ACME Corporation
INVOICE
Invoice Number: INV-2024-0042
Invoice Date: 2024-03-15
Due Date: 2024-04-14
Bill To:
Globex Inc
500 Market Street, San Francisco, CA 94105
Description           Qty    Unit Price    Amount
Consulting services    10        100.00    1000.00
Support plan            1        200.00     200.00
Subtotal:   1200.00
Tax (8%):     96.00
Total Due:  1296.00 USD"""

RESUME_TEXT = """
Jane Smith
jane.smith@example.com | +1 (415) 555-0182
Work Experience
Senior Engineer at Initech (2019 - 2024)
Education
B.S. Computer Science, MIT
Skills: Python, NLP, Docker
"""


def test_classify_invoice():
    doc_type, conf = classify.classify(INVOICE_TEXT)
    assert doc_type == DocumentType.INVOICE
    assert conf > 0.3


def test_classify_resume():
    doc_type, conf = classify.classify(RESUME_TEXT)
    assert doc_type == DocumentType.RESUME
    assert conf > 0.3


def test_grounding_score():
    assert grounding_score("INV-2024-0042", INVOICE_TEXT) > 0.9
    assert grounding_score("NOT-IN-DOC-XYZ", INVOICE_TEXT) < 0.6


def test_field_confidence_monotonic():
    grounded = field_confidence("INV-2024-0042", INVOICE_TEXT, 0.95, corroborated=True)
    ungrounded = field_confidence("ZZZ-9999", INVOICE_TEXT, 0.95, corroborated=False)
    assert grounded > ungrounded


def test_extract_baseline_offline():
    """With no API key, extraction falls back to regex/spaCy and still validates."""
    settings = Settings(openai_api_key="")
    data, fields, warnings = extract.extract_fields(
        doc_type=DocumentType.RESUME,
        text=RESUME_TEXT,
        layout_context=RESUME_TEXT,
        ocr_confidence=0.9,
        settings=settings,
    )
    assert "email" in data
    assert data["email"] == "jane.smith@example.com"
    assert fields["email"].confidence > 0.5


def test_line_item_parsing():
    items = regex_extract.parse_line_items(INVOICE_TABLE_TEXT)
    assert len(items) == 2
    assert items[0]["description"] == "Consulting services"
    assert items[0]["quantity"] == 10.0
    assert items[0]["unit_price"] == 100.0
    assert items[0]["amount"] == 1000.0


def test_labeled_amount_skips_percentages():
    # "Tax (8%): 96.00" must resolve to 96.00, not 8.
    assert regex_extract.labeled_amount(INVOICE_TABLE_TEXT, r"tax") == 96.0
    # "Total Due" must not be confused with "Subtotal".
    assert regex_extract.labeled_amount(
        INVOICE_TABLE_TEXT, r"total\s*due|\btotal\b"
    ) == 1296.0


def test_invoice_baseline_offline():
    settings = Settings(openai_api_key="")
    data, fields, warnings = extract.extract_fields(
        doc_type=DocumentType.INVOICE,
        text=INVOICE_TABLE_TEXT,
        layout_context=INVOICE_TABLE_TEXT,
        ocr_confidence=0.95,
        settings=settings,
    )
    assert data["invoice_number"] == "INV-2024-0042"
    assert data["total"] == 1296.0
    assert data["tax"] == 96.0
    assert data["currency"] == "USD"
    assert data["vendor_name"] == "ACME Corporation"
    assert len(data["items"]) == 2
