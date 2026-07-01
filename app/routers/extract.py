"""API routes for document extraction."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import get_settings
from app.pipeline.orchestrator import process_document
from app.schemas.documents import DocumentType
from app.schemas.response import ExtractionResponse

router = APIRouter(prefix="/api", tags=["extraction"])

_ALLOWED_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


@router.get("/document-types")
def document_types() -> dict:
    return {"types": [t.value for t in DocumentType if t != DocumentType.UNKNOWN]}


@router.post("/extract", response_model=ExtractionResponse)
async def extract_document(
    file: UploadFile = File(...),
    document_type: str | None = Form(default=None),
    include_raw_text: bool = Form(default=True),
) -> ExtractionResponse:
    settings = get_settings()

    filename = file.filename or "upload"
    if not filename.lower().endswith(_ALLOWED_EXT):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Allowed: {', '.join(_ALLOWED_EXT)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.max_upload_mb} MB.",
        )

    forced_type: DocumentType | None = None
    if document_type:
        try:
            forced_type = DocumentType(document_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown document_type: {document_type}")

    try:
        return process_document(
            content=content,
            filename=filename,
            settings=settings,
            forced_type=forced_type,
            include_raw_text=include_raw_text,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Extraction error: {exc}")
