# Understanding the Platform — From Scratch

A learning + reference guide to the **Multi-Document Intelligent Extraction Platform**.
Read this top-to-bottom once; afterwards use it to refresh before interviews or when
extending the code.

---

## 1. The 30-second mental model

> You feed in a messy document (PDF/image). The system **reads** it, figures out **what
> kind** of document it is, **pulls out** the important fields into clean JSON,
> **validates** that JSON against a strict schema, and attaches a **confidence score** to
> every field so a human knows what to double-check.

This category is called **Document AI** / **Intelligent Document Processing (IDP)** — the
same thing sold by AWS Textract, Google Document AI, Rossum, and Docsumo.

The core problem: documents are *unstructured* (pixels and free text), but software needs
*structured* data (typed fields). This project is the bridge.

---

## 2. The pipeline (how data flows)

```
PDF / image
   │
   ▼
[1] OCR ──────────►  raw text + word boxes + per-word confidence
   │
   ▼
[2] Layout analysis ►  text reassembled in reading order, grouped into blocks
   │
   ▼
[3] Classification ►  "this is an invoice" (+ how sure)
   │
   ▼
[4] Extraction ────►  spaCy/regex baseline  ⨁  LLM  ──► raw field dict
   │
   ▼
[5] Validation ────►  coerced into a typed Pydantic model (drops junk, fixes types)
   │
   ▼
[6] Confidence ────►  0–1 score per field; low ones flagged "needs_review"
   │
   ▼
Structured, scored JSON
```

Each numbered stage is one file in `app/pipeline/`. `orchestrator.py` is the conductor
that calls them in order — **read that file first** when you revisit the project; it's the
table of contents.

---

## 3. Stage-by-stage walkthrough

Each stage below names the **AI concept** it demonstrates and the **file** that implements it.

### Stage 1 — OCR · `app/pipeline/ocr.py`
**Concept: Optical Character Recognition.** Turns *pixels of text* into *characters*.

- **PDF with a real text layer** → read directly with `pdfplumber` (fast + lossless, no OCR).
- **Image or scanned PDF** → run **Tesseract**, which returns each word plus its bounding
  box `(left, top, width, height)` and a **per-word confidence** (0–1).
- Word boxes are kept because the next two stages need geometry, and confidence feeds the
  final scoring.

> Trade-off worth mentioning: hybrid OCR (native text when available, Tesseract fallback
> for scans) maximizes both accuracy and speed.

### Stage 2 — Layout analysis · `app/pipeline/layout.py`
**Concept: Document AI / spatial layout analysis.** Raw OCR output can be jumbled
(multi-column docs especially). Using word **coordinates**, this stage:

- groups words into **lines** (similar vertical position),
- groups lines into **blocks** (separated by vertical gaps),
- produces text in correct **reading order**.

This is **not** currently a trained LayoutParser/Detectron2 model. It is a lightweight
geometric implementation inspired by LayoutParser-style document structure modeling. The
important design decision is that it returns a clean `LayoutResult` shape, so a full
LayoutParser, Detectron2, PubLayNet, or LayoutLM-based layout model can be plugged in later.

### Stage 3 — Classification · `app/pipeline/classify.py`
**Concept: document classification.** You must know *what* to extract before extracting
(an invoice has `invoice_number`; a resume has `skills`). This is a **weighted keyword
classifier**: each document type has signal phrases with weights (e.g. `"purchase order"`
= 4.0), it scores all six types, picks the highest, and derives confidence from
"share of evidence" + "absolute strength." Fast, explainable, fully offline.

This is intentionally simple for the MVP. In an interview, present it honestly as a baseline.
The obvious upgrade is sentence-transformer embeddings, LayoutLM, or a fine-tuned document
classifier once a labeled dataset exists.

### Stage 4 — Extraction · `extract.py`, `llm_extract.py`, `spacy_ner.py`, `regex_extract.py`
**Concepts: NER + LLM extraction + fusion.** This is the heart. Two independent extractors run:

- **Baseline (local, no API):** **spaCy NER** finds entities (PERSON, ORG, GPE, DATE…), and
  **regex** pulls structured patterns (emails, phones, money, dates, reference numbers,
  line-item tables). Deterministic and free.
- **LLM (OpenAI):** the model receives the document text *plus the target JSON schema* and
  fills it in. Uses `response_format={"type": "json_object"}` and `temperature=0` so output
  is deterministic, parseable JSON.

**Fusion** is the clever part: the LLM proposes values, and regex/spaCy **corroborate** them.
If both independently agree, the field's source is marked `"fusion"` and confidence is
boosted. If the LLM is unavailable, the baseline carries the whole load — which is why it's
built to be strong on its own.

> Concept names to use: *Named Entity Recognition (NER)*, *schema-guided / structured LLM
> extraction*, *prompt engineering with JSON-mode enforcement*, *ensemble/fusion of
> deterministic + probabilistic extractors*.

### Stage 5 — Validation · `app/schemas/documents.py`
**Concept: schema validation / typed contracts.** Each document type is a **Pydantic**
model (`Invoice`, `Receipt`, `PurchaseOrder`, `Resume`, `Contract`, `IdDocument`). The raw
dict from stage 4 goes through `model_validate()`, which:

- drops unknown keys,
- coerces types (string `"1296.00"` → float `1296.0`),
- guarantees the output always matches the contract (every field optional, so partial
  extractions still validate instead of crashing).

This is what makes the output *trustworthy for downstream software*.

### Stage 6 — Confidence scoring · `app/pipeline/confidence.py`
**Concept: confidence scoring + human-in-the-loop.** LLMs don't give calibrated
probabilities, so confidence is derived from measurable signals:

- **grounding** — does the value actually appear in the source text? (fuzzy-matched to
  tolerate OCR noise)
- **corroboration** — did the independent baseline find the same value?
- **OCR quality** — mean per-word OCR confidence for the document.

These blend into a 0–1 score. Fields below a threshold are flagged in `needs_review` — the
human-in-the-loop routing that real IDP systems require.

### Serving layer · `main.py`, `routers/extract.py`, `static/index.html`
**FastAPI** exposes `POST /api/extract` (upload a file → get JSON), auto-generates Swagger
docs at `/docs`, and serves a drag-and-drop web UI. The response envelope
(`schemas/response.py`) bundles data, per-field confidence, OCR metadata, and warnings.

---

## 4. Project layout

```
app/
  main.py                FastAPI app + web UI route + /health
  config.py              env-based settings (.env)
  schemas/
    documents.py         Pydantic models per doc type + SCHEMA_REGISTRY
    response.py          API response envelope (data + per-field confidence)
  routers/
    extract.py           POST /api/extract, GET /api/document-types
  pipeline/
    ocr.py               Tesseract / pdfplumber OCR (text + word boxes + confidence)
    layout.py            geometric layout analysis (pluggable layout-model interface)
    classify.py          weighted-keyword document-type classifier
    spacy_ner.py         spaCy NER baseline
    regex_extract.py     deterministic extractors (money, dates, ids, line items...)
    llm_extract.py       OpenAI schema-guided structured extraction
    confidence.py        grounding-based confidence scoring
    extract.py           fusion of LLM + baseline, validation, scoring
    orchestrator.py      end-to-end pipeline (the conductor)
  static/index.html      drag-and-drop web UI
tests/                   offline pipeline tests (no OCR binary or API key needed)
scripts/make_sample.py   generates a sample invoice image
scripts/evaluate.py      computes field-level benchmark metrics
data/benchmark.jsonl     small labeled synthetic benchmark
```

---

## 5. Running it

### One-time setup
```bash
# System binaries (macOS)
brew install tesseract poppler

# Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Config (optional — leave key blank to run fully offline)
cp .env.example .env   # set OPENAI_API_KEY to enable LLM mode
```

### Run the server
```bash
uvicorn app.main:app --reload
```
- Web UI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

### Try it
```bash
python scripts/make_sample.py
curl -s -X POST http://127.0.0.1:8000/api/extract \
  -F "file=@samples/invoice.png" | python -m json.tool
```

### Tests
```bash
pytest -q          # runs fully offline
```

### Evaluation
```bash
python scripts/evaluate.py
```

The included benchmark is intentionally small and synthetic. Current offline-baseline result:

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

Use those numbers as a development sanity check, not as production claims. For resume-grade
metrics, expand the benchmark with public or manually labeled documents: CORD for receipts,
FUNSD for forms, and additional invoices/resumes/purchase orders with blur, skew, rotation,
different layouts, and OCR noise.

### Two modes
- **LLM mode** (set `OPENAI_API_KEY`): LLM extracts into the schema; regex/spaCy corroborate.
- **Offline mode** (blank key): spaCy + regex baseline only — no external calls.

---

## 6. Why the design choices are good

- **Pluggable stages** — swap Tesseract↔PaddleOCR or geometric↔LayoutParser/Detectron2 without
  touching the rest. Clean separation of concerns.
- **Graceful degradation** — works fully offline; the LLM is an enhancement, not a hard
  dependency. Bad files return warnings instead of crashing.
- **Trust, not just extraction** — most demos stop at "LLM returns JSON." This adds
  validation + confidence + review routing, which makes it *production-shaped*.
- **Deterministic + probabilistic ensemble** — combining regex/spaCy with an LLM is more
  robust and more explainable than either alone.

---

## 7. Interview Q&A

- **"How do you trust LLM output?"** — Validate against a Pydantic schema, ground each value
  back in the source text, corroborate with independent regex/spaCy extractors, and score
  confidence so low-trust fields get routed to human review.
- **"What if there's no internet / API key?"** — Degrades gracefully to a local spaCy+regex
  baseline; the LLM is an enhancer, not a hard dependency.
- **"How do you handle different document types?"** — A keyword classifier routes to one of
  six typed schemas; the extractor is schema-driven, so adding a type = add a Pydantic model
  + keywords. This is the MVP baseline; a stronger version would use sentence embeddings or
  LayoutLM on labeled examples.
- **"How does confidence work?"** — A blend of grounding (fuzzy match to OCR text),
  corroboration, and OCR word confidence, because LLMs aren't calibrated.
- **"How would you scale / productionize?"** — Async workers/queue for OCR, a batch
  endpoint, caching, swap in real LayoutParser + PaddleOCR, and an eval harness with labeled
  data reporting per-field F1.
- **"Are you using real LayoutParser?"** — No. The current implementation is geometric layout
  reconstruction from OCR word boxes. It is inspired by LayoutParser-style structure modeling
  and designed so a real LayoutParser/Detectron2 model can replace it behind the same interface.

---

## 8. Extending the project

- **Add a document type:** add a Pydantic model in `app/schemas/documents.py`, register it in
  `SCHEMA_REGISTRY`, and add keyword signals in `app/pipeline/classify.py`.
- **Swap OCR engine (e.g. PaddleOCR):** implement an alternative in `app/pipeline/ocr.py`
  returning the same `OcrResult` shape.
- **Real LayoutParser:** replace `analyze_layout` in `app/pipeline/layout.py` with a
  Detectron2-backed version returning the same `LayoutResult`.
- **Evaluation:** add labeled examples and a script computing per-field precision/recall/F1
  to quantify accuracy over time.

---

## 9. Limitations and roadmap

- **Layout:** current layout parsing is geometric, not a trained LayoutParser/Detectron2 model.
  Roadmap: integrate LayoutParser with PubLayNet/FUNSD-style layout labels.
- **Classifier:** current classifier is keyword based. Roadmap: sentence-transformer cosine
  similarity for a quick upgrade, then LayoutLM/fine-tuned classifier for document images.
- **Metrics:** the included benchmark is small and synthetic. Roadmap: evaluate on CORD, FUNSD,
  and a larger set of labeled invoices/resumes/receipts before quoting resume metrics.
- **OCR:** Tesseract struggles with handwriting, low-resolution images, rotation, and noisy
  receipts. Roadmap: add preprocessing and optionally PaddleOCR.
- **Tables:** line-item extraction is heuristic. Roadmap: table detection/structure recovery
  using layout models or table-specific parsers.

---

## 10. Glossary

- **OCR** — Optical Character Recognition; images/scans → text.
- **NER** — Named Entity Recognition; tagging spans as PERSON, ORG, DATE, etc.
- **Document AI / IDP** — turning documents into structured, machine-usable data.
- **Layout analysis** — recovering the spatial/reading structure of a page.
- **Schema-guided extraction** — giving the LLM a target JSON schema to fill.
- **Grounding** — checking an extracted value actually appears in the source text.
- **Human-in-the-loop** — routing low-confidence results to a human reviewer.
