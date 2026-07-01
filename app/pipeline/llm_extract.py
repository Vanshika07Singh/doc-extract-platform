"""LLM-based structured extraction via the OpenAI API.

The target schema (derived from the Pydantic model) is sent to the model with
``response_format`` JSON-schema enforcement so the reply is guaranteed-parseable
JSON shaped like the document model.
"""

from __future__ import annotations

import json
from typing import Optional

from app.schemas.documents import DocumentType, json_schema_for

_SYSTEM_PROMPT = (
    "You are a precise document information extraction engine. "
    "Extract the requested fields from the document text. "
    "Rules:\n"
    "- Only use information present in the text; never invent values.\n"
    "- If a field is missing, set it to null (or an empty list for arrays).\n"
    "- Normalize all dates to ISO 8601 (YYYY-MM-DD) when possible.\n"
    "- Return numbers for monetary/quantity fields without currency symbols.\n"
    "- Preserve currency as a separate ISO 4217 code when available."
)


def _build_user_prompt(doc_type: DocumentType, text: str, layout_context: str) -> str:
    schema = json.dumps(json_schema_for(doc_type), indent=2)
    context = layout_context.strip() or text
    return (
        f"Document type: {doc_type.value}\n\n"
        f"Target JSON schema:\n{schema}\n\n"
        "Document text (OCR / parsed):\n"
        "\"\"\"\n"
        f"{context[:15000]}\n"
        "\"\"\"\n\n"
        "Return ONLY a JSON object matching the schema."
    )


def llm_extract(
    doc_type: DocumentType,
    text: str,
    layout_context: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Optional[dict]:
    """Call the LLM and return a parsed dict, or None on failure."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(doc_type, text, layout_context),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as exc:  # noqa: BLE001 - surfaced as a warning upstream
        return {"__error__": str(exc)}
