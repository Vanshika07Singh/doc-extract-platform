"""Deterministic regex extractors used to corroborate LLM output."""

from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s-]?)?(?:\(\d{1,4}\)[\s-]?)?\d[\d\s().-]{6,}\d)")
# Money like $1,234.56 / 1234.56 USD / €99
MONEY_RE = re.compile(
    r"(?:[$€£₹]\s?\d{1,3}(?:[,\d]{0,})(?:\.\d{2})?)|(?:\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b)"
)
DATE_RE = re.compile(
    r"\b("
    r"\d{4}-\d{2}-\d{2}"  # 2024-01-31
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"  # 31/01/2024
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r")\b",
    re.IGNORECASE,
)


NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}


def find_emails(text: str) -> list[str]:
    return EMAIL_RE.findall(text)


def find_phones(text: str) -> list[str]:
    return [m.strip() for m in PHONE_RE.findall(text) if sum(c.isdigit() for c in m) >= 7]


def find_money(text: str) -> list[str]:
    return [m.strip() for m in MONEY_RE.findall(text)]


def find_dates(text: str) -> list[str]:
    return [m if isinstance(m, str) else m[0] for m in DATE_RE.findall(text)]


def to_float(token: str | None) -> float | None:
    """Parse a numeric token like '1,296.00' into a float."""
    if not token:
        return None
    m = NUMBER_RE.search(token)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def labeled_value(text: str, label_pattern: str) -> str | None:
    """Return the text that follows ``label[: ]`` on the same line, if any."""
    rx = re.compile(rf"(?:{label_pattern})\s*[:#\-]?\s*(.+)", re.IGNORECASE)
    for line in text.splitlines():
        m = rx.search(line)
        if m:
            val = m.group(1).strip(" :#-\t")
            if val:
                return val
    return None


_BLOCK_STOP_RE = re.compile(
    r"\b(description|qty|quantity|unit\s*price|sub\s*total|total|tax|amount|invoice|date)\b",
    re.IGNORECASE,
)


def labeled_block(text: str, label_pattern: str, max_lines: int = 2) -> str | None:
    """Return a value that may continue onto the lines below the label.

    Useful for blocks like ``Bill To:`` where the address is on the next lines.
    Stops early at section headers (e.g. an items table) so it doesn't bleed.
    """
    lines = text.splitlines()
    rx = re.compile(rf"(?:{label_pattern})\s*[:#\-]?\s*(.*)", re.IGNORECASE)
    for i, line in enumerate(lines):
        m = rx.search(line)
        if m:
            same = m.group(1).strip(" :#-\t")
            if same:
                return same
            collected: list[str] = []
            for nxt in lines[i + 1 : i + 1 + max_lines]:
                if not nxt.strip():
                    if collected:
                        break
                    continue
                if _BLOCK_STOP_RE.search(nxt):
                    break
                collected.append(nxt.strip())
            return ", ".join(collected) or None
    return None


def labeled_amount(text: str, label_pattern: str) -> float | None:
    """Return the monetary amount following a label (e.g. 'Total Due: 1296.00').

    Picks the right-most number that isn't a percentage, so values like
    'Tax (8%): 96.00' resolve to 96.00 rather than 8.
    """
    val = labeled_value(text, label_pattern)
    if not val:
        return None
    candidates: list[str] = []
    for m in NUMBER_RE.finditer(val):
        if val[m.end() : m.end() + 1] == "%":
            continue
        candidates.append(m.group(0))
    return to_float(candidates[-1]) if candidates else None


def labeled_date(text: str, label_pattern: str) -> str | None:
    """Return the first date found on the same line as a label."""
    val = labeled_value(text, label_pattern)
    if val:
        dates = find_dates(val)
        if dates:
            return dates[0]
    return None


def find_reference_number(text: str, label_pattern: str) -> str | None:
    """Return an identifier (e.g. INV-2024-0042) following a label.

    Tolerates OCR noise that inserts spaces around separators, so a value read
    as ``INV- 2024 -0042`` is normalised back to ``INV-2024-0042``.
    """
    val = labeled_value(text, label_pattern)
    if not val:
        return None
    # Normalise OCR'd unicode dashes (en/em/figure dashes) to ASCII hyphen.
    val = re.sub(r"[\u2010-\u2015\u2212]", "-", val)
    # Identifier possibly broken by OCR spaces around '-' / '/'.
    m = re.search(
        r"[A-Za-z0-9]+(?:\s*[-/_]\s*[A-Za-z0-9]+)+|[A-Za-z0-9][A-Za-z0-9\-/_]{2,}",
        val,
    )
    if not m:
        return val.split()[0]
    return re.sub(r"\s+", "", m.group(0))


def find_currency(text: str) -> str | None:
    """Detect an ISO currency code or symbol in the text."""
    m = re.search(r"\b(USD|EUR|GBP|INR|JPY|CAD|AUD|CHF|CNY)\b", text)
    if m:
        return m.group(1).upper()
    for sym, code in CURRENCY_SYMBOLS.items():
        if sym in text:
            return code
    return None


def parse_line_items(text: str) -> list[dict]:
    """Heuristically parse tabular line items: a description followed by qty,
    unit price, and amount (the most common 3-number invoice/PO/receipt layout).
    """
    num = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
    row_rx = re.compile(
        rf"^(?P<desc>.*?[A-Za-z].*?)\s+(?P<qty>{num})\s+(?P<unit>{num})\s+(?P<amount>{num})\s*$"
    )
    skip_rx = re.compile(
        r"\b(sub\s*total|total|tax|vat|gst|amount\s*due|balance|qty|unit\s*price|description)\b",
        re.IGNORECASE,
    )
    items: list[dict] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line or skip_rx.search(line):
            continue
        m = row_rx.match(line)
        if not m:
            continue
        items.append(
            {
                "description": m.group("desc").strip(),
                "quantity": to_float(m.group("qty")),
                "unit_price": to_float(m.group("unit")),
                "amount": to_float(m.group("amount")),
            }
        )
    return items
