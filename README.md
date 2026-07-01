# Multi-Document Intelligent Extraction Platform

Turn messy documents — invoices, receipts, purchase orders, resumes, contracts, and IDs — into clean, validated, **confidence-scored JSON**.

```
PDF / image  →  OCR  →  Geometric layout analysis  →  Classification  →  NER + LLM extraction  →  Pydantic validation  →  Scored JSON
```

## What it does

- **OCR** — native PDF text via `pdfplumber`, scanned pages & images via **Tesseract** (with word-level boxes + confidence).
- **Layout analysis** — lightweight geometric reconstruction of lines/blocks/reading order, inspired by LayoutParser-style document structure modeling. The code is pluggable so full layout models can be added later.
- **Classification** — fast, explainable keyword-scored document-type detection. This is intentionally simple for the MVP and documented as a limitation.
- **Extraction** — **OpenAI LLM** guided by a JSON schema, corroborated by a **spaCy + regex** baseline.
- **Validation** — every output is coerced into a typed **Pydantic** model per document type.
- **Confidence scoring** — each field gets a 0–1 score from *grounding* (does the value appear in the OCR text?), *corroboration* (did regex/spaCy independently find it?), and *OCR quality*. Low-confidence fields are flagged for review.

### Example output

```json
{
  "document_type": "invoice",
  "type_confidence": 0.82,
  "data": {
    "invoice_number": "INV-2024-0042",
    "billing_address": "Globex Inc, 500 Market Street, San Francisco, CA 94105",
    "total": 1296.00,
    "currency": "USD",
    "items": [{ "description": "Consulting services", "quantity": 10, "unit_price": 100.0, "amount": 1000.0 }]
  },
  "fields": {
    "invoice_number": { "value": "INV-2024-0042", "confidence": 0.94, "source": "fusion", "needs_review": false }
  },
  "overall_confidence": 0.88,
  "needs_review": [],
  "ocr": { "engine": "tesseract", "pages": 1, "mean_word_confidence": 0.91, "char_count": 412 }
}
```

## Project layout

```
app/
  main.py                FastAPI app + web UI route
  config.py              env-based settings
  schemas/               Pydantic models (per doc type) + response envelope
  pipeline/
    ocr.py               Tesseract / pdfplumber OCR
    layout.py            geometric layout analysis
    classify.py          document-type classifier
    spacy_ner.py         spaCy NER baseline
    regex_extract.py     deterministic extractors
    llm_extract.py       OpenAI structured extraction
    confidence.py        grounding-based confidence scoring
    extract.py           fusion + validation
    orchestrator.py      end-to-end pipeline
  static/index.html      drag-and-drop web UI
tests/                   offline pipeline tests
scripts/make_sample.py   generates a sample invoice image
scripts/evaluate.py      computes field-level benchmark metrics
data/benchmark.jsonl     small labeled synthetic benchmark
```

## Setup

### 1. System dependencies

OCR needs the Tesseract binary, and scanned-PDF support needs Poppler:

```bash
# macOS
brew install tesseract poppler

# Ubuntu/Debian
sudo apt-get install -y tesseract-ocr poppler-utils
```

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# spaCy English model (used by the baseline extractor)
python -m spacy download en_core_web_sm
```

### 3. Configure

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY (leave blank to run offline in spaCy-only mode)
```

## Run

```bash
uvicorn app.main:app --reload
```

- Web UI: http://127.0.0.1:8000
- API docs (Swagger): http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

### Try it via the API

```bash
# generate a sample invoice image first
python scripts/make_sample.py

curl -s -X POST http://127.0.0.1:8000/api/extract \
  -F "file=@samples/invoice.png" | python -m json.tool
```

You can force a document type instead of auto-detecting:

```bash
curl -s -X POST http://127.0.0.1:8000/api/extract \
  -F "file=@samples/invoice.png" -F "document_type=invoice"
```

## Tests

The included tests run fully offline (no Tesseract or API key required):

```bash
pip install pytest
pytest -q
```

## Evaluation

The repo includes a small labeled **synthetic** benchmark so extraction quality can be measured instead of only demonstrated:

```bash
python scripts/evaluate.py
```

Current offline-baseline result on `data/benchmark.jsonl`:

```json
{
  "documents": 6,
  "classification_accuracy": 1.0,
  "field_accuracy": 0.9706,
  "precision": 0.825,
  "recall": 0.9706,
  "f1": 0.8919
}
```

These numbers are useful for development sanity checks, but they should **not** be presented as production accuracy. For resume-grade metrics, expand the benchmark with public datasets such as CORD (receipts), FUNSD (forms), or a labeled set of invoices/resumes/POs with varied layouts, scans, blur, rotation, and OCR noise.

## Modes

- **LLM mode** (recommended): set `OPENAI_API_KEY`. The LLM extracts into the schema; regex/spaCy corroborate to raise confidence.
- **Offline mode**: leave the key blank. Extraction uses spaCy + regex only — fewer fields, lower confidence, but no external calls.

## Extending

- **Add a document type**: add a Pydantic model in `app/schemas/documents.py`, register it in `SCHEMA_REGISTRY`, and add keyword signals in `app/pipeline/classify.py`.
- **Swap OCR engine** (e.g. PaddleOCR): implement an alternative in `app/pipeline/ocr.py` returning the same `OcrResult` shape.
- **Full layout models**: replace `analyze_layout` in `app/pipeline/layout.py` with a LayoutParser/Detectron2-backed version returning the same `LayoutResult`.
- **Stronger classification**: replace keyword scoring with sentence-transformer embeddings, LayoutLM, or a fine-tuned document classifier once a labeled dataset exists.

## Limitations

- The current layout stage is **geometric**, not a trained LayoutParser/Detectron2 model.
- The document-type classifier is rule/keyword based, which is explainable but weaker than embedding or transformer-based classifiers on ambiguous documents.
- The included benchmark is small and synthetic. It demonstrates the evaluation loop but is not enough to claim real-world accuracy.
- OCR quality can drop on handwriting, low-resolution scans, heavy skew/rotation, or noisy receipts.
- Table extraction is heuristic and works best for simple rows with description, quantity, unit price, and amount.

## AI concepts demonstrated

OCR · Named Entity Recognition · Document AI / geometric layout analysis · LLM structured extraction · schema validation · confidence scoring & human-in-the-loop routing · evaluation metrics.
