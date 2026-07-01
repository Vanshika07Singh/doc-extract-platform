"""Typed, validated output schemas for each supported document type.

These models define the *target structure* the pipeline extracts into.
Every field is Optional so that partial extractions still validate; missing
data is represented as ``null`` rather than raising.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Type

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    PURCHASE_ORDER = "purchase_order"
    RESUME = "resume"
    CONTRACT = "contract"
    ID_DOCUMENT = "id_document"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Shared sub-models
# --------------------------------------------------------------------------- #
class LineItem(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None


class Money(BaseModel):
    amount: Optional[float] = None
    currency: Optional[str] = Field(default=None, description="ISO 4217 code, e.g. USD")


# --------------------------------------------------------------------------- #
# Document schemas
# --------------------------------------------------------------------------- #
class Invoice(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = Field(default=None, description="ISO 8601 date")
    due_date: Optional[str] = Field(default=None, description="ISO 8601 date")
    vendor_name: Optional[str] = None
    billing_address: Optional[str] = None
    shipping_address: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    items: list[LineItem] = Field(default_factory=list)


class Receipt(BaseModel):
    merchant_name: Optional[str] = None
    transaction_date: Optional[str] = Field(default=None, description="ISO 8601 date")
    payment_method: Optional[str] = None
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    items: list[LineItem] = Field(default_factory=list)


class PurchaseOrder(BaseModel):
    po_number: Optional[str] = None
    order_date: Optional[str] = Field(default=None, description="ISO 8601 date")
    buyer_name: Optional[str] = None
    supplier_name: Optional[str] = None
    shipping_address: Optional[str] = None
    total: Optional[float] = None
    currency: Optional[str] = None
    items: list[LineItem] = Field(default_factory=list)


class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_year: Optional[str] = None
    end_year: Optional[str] = None


class Experience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    summary: Optional[str] = None


class Resume(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)


class Contract(BaseModel):
    title: Optional[str] = None
    parties: list[str] = Field(default_factory=list)
    effective_date: Optional[str] = Field(default=None, description="ISO 8601 date")
    expiration_date: Optional[str] = Field(default=None, description="ISO 8601 date")
    governing_law: Optional[str] = None
    total_value: Optional[float] = None
    currency: Optional[str] = None
    key_terms: list[str] = Field(default_factory=list)


class IdDocument(BaseModel):
    document_kind: Optional[str] = Field(
        default=None, description="e.g. passport, driver_license, national_id"
    )
    full_name: Optional[str] = None
    document_number: Optional[str] = None
    date_of_birth: Optional[str] = Field(default=None, description="ISO 8601 date")
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    nationality: Optional[str] = None
    issuing_authority: Optional[str] = None


SCHEMA_REGISTRY: dict[DocumentType, Type[BaseModel]] = {
    DocumentType.INVOICE: Invoice,
    DocumentType.RECEIPT: Receipt,
    DocumentType.PURCHASE_ORDER: PurchaseOrder,
    DocumentType.RESUME: Resume,
    DocumentType.CONTRACT: Contract,
    DocumentType.ID_DOCUMENT: IdDocument,
}


def schema_for(doc_type: DocumentType) -> Type[BaseModel]:
    """Return the Pydantic model class for a document type (defaults to Invoice)."""
    return SCHEMA_REGISTRY.get(doc_type, Invoice)


def json_schema_for(doc_type: DocumentType) -> dict:
    """Return the JSON schema dict used to steer the LLM extraction."""
    return schema_for(doc_type).model_json_schema()
