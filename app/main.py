"""FastAPI entrypoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.routers import extract as extract_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Multi-Document Intelligent Extraction Platform",
    version=__version__,
    description="OCR + layout + NER/LLM extraction with confidence scoring.",
)

app.include_router(extract_router.router)


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "llm_enabled": settings.llm_enabled,
        "ocr_lang": settings.ocr_lang,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
