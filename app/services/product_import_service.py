from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, status

from app.schemas.imports import ImportPreviewResponse, ImportedProductRow
from app.utils.product_text import build_category, infer_product_type, normalize_title

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover
    PdfReader = None


@dataclass(slots=True)
class ParsedImportPayload:
    filename: str
    source_type: str
    rows: list[ImportedProductRow]


class ProductImportService:
    TITLE_COLUMNS = ("title", "name", "product", "product_name", "item-name", "item_name")
    SKU_COLUMNS = ("sku", "seller_sku", "item_sku", "product_id", "product-id")
    BRAND_COLUMNS = ("brand", "brand_name", "brand-name", "vendor", "manufacturer")
    CATEGORY_COLUMNS = ("category", "product_category", "department", "feed-product-type")
    PRODUCT_TYPE_COLUMNS = ("product_type", "type", "item_type")
    DESCRIPTION_COLUMNS = ("description", "product_description", "body_html", "details")
    PRICE_COLUMNS = ("price", "standard-price", "amount", "sale_price", "list_price")
    QUANTITY_COLUMNS = ("quantity", "qty", "stock", "available", "inventory")
    COLOR_COLUMNS = ("color", "color_name", "color-name")
    SIZE_COLUMNS = ("size", "size_name", "size-name")
    MATERIAL_COLUMNS = ("material", "material_type", "material-type")
    IMAGE_COLUMNS = ("image_url", "image", "main-image-url", "main_image_url", "featured_image")

    def parse_file(self, *, filename: str, payload: bytes, existing_titles: set[str] | None = None) -> ImportPreviewResponse:
        source_type = self._detect_source_type(filename)
        if source_type == "csv":
            rows = self._parse_csv(filename, payload)
        elif source_type == "excel":
            rows = self._parse_excel(filename, payload)
        else:
            rows = self._parse_pdf(filename, payload)

        self._mark_duplicates(rows, existing_titles or set())
        return ImportPreviewResponse(
            filename=filename,
            source_type=source_type,
            total_rows=len(rows),
            ready_rows=sum(1 for row in rows if row.status == "ready"),
            duplicate_rows=sum(1 for row in rows if row.status == "duplicate"),
            parse_issue_rows=sum(1 for row in rows if row.status == "parse_issue"),
            rows=rows,
        )

    @staticmethod
    def _detect_source_type(filename: str) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in {".xlsx", ".xls"}:
            return "excel"
        if suffix == ".pdf":
            return "pdf"
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported import file. Use CSV, Excel, or PDF.",
        )

    def _parse_csv(self, filename: str, payload: bytes) -> list[ImportedProductRow]:
        text = payload.decode("utf-8-sig", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames is None:
            return []
        rows = [self._normalize_record(index, record, source_type="csv") for index, record in enumerate(reader, start=1)]
        return [row for row in rows if any(self._row_has_data(row))]

    def _parse_excel(self, filename: str, payload: bytes) -> list[ImportedProductRow]:
        try:
            frame = pd.read_excel(io.BytesIO(payload)).fillna("")
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not read Excel file: {exc}",
            ) from exc
        rows = [
            self._normalize_record(index, {str(key): value for key, value in row.items()}, source_type="excel")
            for index, row in enumerate(frame.to_dict(orient="records"), start=1)
        ]
        return [row for row in rows if any(self._row_has_data(row))]

    def _parse_pdf(self, filename: str, payload: bytes) -> list[ImportedProductRow]:
        if PdfReader is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF import requires pypdf to be installed.",
            )
        try:
            reader = PdfReader(io.BytesIO(payload))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Could not read PDF file: {exc}",
            ) from exc

        extracted_pages = [(page_number, (page.extract_text() or "").strip()) for page_number, page in enumerate(reader.pages, start=1)]
        raw_blocks: list[tuple[int, str]] = []
        for page_number, text in extracted_pages:
            if not text:
                continue
            for block in re.split(r"\n\s*\n+", text):
                cleaned = block.strip()
                if cleaned:
                    raw_blocks.append((page_number, cleaned))

        if not raw_blocks:
            return [
                ImportedProductRow(
                    row_id="pdf-1",
                    source_type="pdf",
                    source_reference=f"{filename}:page-1",
                    status="parse_issue",
                    confidence=0.0,
                    missing_fields=["title"],
                    notes=["No extractable text was found in this PDF. Scanned PDFs need manual review."],
                )
            ]

        rows = [self._normalize_pdf_block(index, page_number, block) for index, (page_number, block) in enumerate(raw_blocks, start=1)]
        return rows

    def _normalize_record(self, index: int, record: dict[str, Any], *, source_type: str) -> ImportedProductRow:
        title = self._first_value(record, self.TITLE_COLUMNS)
        sku = self._first_value(record, self.SKU_COLUMNS)
        brand = self._first_value(record, self.BRAND_COLUMNS)
        category = self._first_value(record, self.CATEGORY_COLUMNS)
        product_type = self._first_value(record, self.PRODUCT_TYPE_COLUMNS)
        description = self._first_value(record, self.DESCRIPTION_COLUMNS)
        price = self._numeric_string(self._first_value(record, self.PRICE_COLUMNS))
        quantity = self._integer_string(self._first_value(record, self.QUANTITY_COLUMNS))
        color = self._first_value(record, self.COLOR_COLUMNS)
        size = self._first_value(record, self.SIZE_COLUMNS)
        material = self._first_value(record, self.MATERIAL_COLUMNS)
        image_url = self._first_value(record, self.IMAGE_COLUMNS)

        resolved_title = normalize_title(title) if title else ""
        resolved_product_type = product_type or infer_product_type(title, category, description)
        resolved_category = category or build_category(resolved_product_type)
        missing_fields = self._missing_fields(
            title=resolved_title,
            price=price,
            quantity=quantity,
        )
        status_value = "ready" if not missing_fields else "missing_data"
        confidence = self._confidence_for(
            title=resolved_title,
            sku=sku,
            description=description,
            price=price,
            quantity=quantity,
            image_url=image_url,
        )

        return ImportedProductRow(
            row_id=f"{source_type}-{index}",
            source_type=source_type,  # type: ignore[arg-type]
            source_reference=f"row-{index}",
            title=resolved_title,
            sku=sku,
            brand=brand,
            category=resolved_category,
            product_type=resolved_product_type,
            description=description,
            price=price,
            quantity=quantity,
            color=color,
            size=size,
            material=material,
            image_url=image_url,
            status=status_value,  # type: ignore[arg-type]
            confidence=confidence,
            missing_fields=missing_fields,
            notes=[],
        )

    def _normalize_pdf_block(self, index: int, page_number: int, block: str) -> ImportedProductRow:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        text = " ".join(lines)
        title = self._extract_pdf_field(block, r"(?:title|product|item)\s*[:\-]\s*(.+)")
        if not title and lines:
            title = lines[0]
        sku = self._extract_pdf_field(block, r"(?:sku|item sku|seller sku|asin)\s*[:#\-]?\s*([A-Z0-9\-_]+)")
        brand = self._extract_pdf_field(block, r"(?:brand|vendor|manufacturer)\s*[:\-]\s*(.+)")
        category = self._extract_pdf_field(block, r"(?:category|department)\s*[:\-]\s*(.+)")
        description = self._extract_pdf_field(block, r"(?:description|details)\s*[:\-]\s*(.+)")
        price = self._extract_currency(block)
        quantity = self._extract_quantity(block)
        color = self._extract_pdf_field(block, r"(?:color|colour)\s*[:\-]\s*(.+)")
        size = self._extract_pdf_field(block, r"(?:size)\s*[:\-]\s*(.+)")
        material = self._extract_pdf_field(block, r"(?:material)\s*[:\-]\s*(.+)")
        image_url = self._extract_pdf_field(block, r"(https?://\S+\.(?:png|jpg|jpeg|webp))")

        resolved_title = normalize_title(title) if title else ""
        resolved_product_type = infer_product_type(resolved_title, category, description, text)
        resolved_category = category or build_category(resolved_product_type)
        missing_fields = self._missing_fields(title=resolved_title, price=price, quantity=quantity)
        notes: list[str] = []
        if not title:
            notes.append("PDF block did not contain an explicit title; first line was used when available.")
        if not block.strip():
            notes.append("Empty PDF block.")
        confidence = self._confidence_for(
            title=resolved_title,
            sku=sku,
            description=description,
            price=price,
            quantity=quantity,
            image_url=image_url,
        )
        status_value = "ready" if resolved_title else "parse_issue"
        if resolved_title and missing_fields:
            status_value = "missing_data"

        return ImportedProductRow(
            row_id=f"pdf-{index}",
            source_type="pdf",
            source_reference=f"page-{page_number}",
            title=resolved_title,
            sku=sku,
            brand=brand,
            category=resolved_category,
            product_type=resolved_product_type,
            description=description or text[:1000],
            price=price,
            quantity=quantity,
            color=color,
            size=size,
            material=material,
            image_url=image_url,
            status=status_value,  # type: ignore[arg-type]
            confidence=confidence,
            missing_fields=missing_fields,
            notes=notes,
        )

    def _mark_duplicates(self, rows: list[ImportedProductRow], existing_titles: set[str]) -> None:
        sku_counts = Counter(row.sku.strip().lower() for row in rows if row.sku.strip())
        title_counts = Counter(row.title.strip().lower() for row in rows if row.title.strip())
        for row in rows:
            duplicate = False
            if row.sku and sku_counts[row.sku.strip().lower()] > 1:
                duplicate = True
                row.notes.append("Duplicate SKU found in uploaded file.")
            normalized_title = row.title.strip().lower()
            if normalized_title and title_counts[normalized_title] > 1:
                duplicate = True
                row.notes.append("Duplicate title found in uploaded file.")
            if normalized_title and normalized_title in existing_titles:
                duplicate = True
                row.notes.append("A product with the same title already exists in the catalog.")
            if duplicate and row.status != "parse_issue":
                row.status = "duplicate"

    @staticmethod
    def _row_has_data(row: ImportedProductRow) -> tuple[bool, ...]:
        return (
            bool(row.title),
            bool(row.sku),
            bool(row.description),
            bool(row.price),
            bool(row.quantity),
            bool(row.image_url),
        )

    @staticmethod
    def _find_key(record: dict[str, Any], candidates: tuple[str, ...]) -> str | None:
        lowered = {str(key).strip().lower(): key for key in record.keys()}
        for candidate in candidates:
            match = lowered.get(candidate.lower())
            if match is not None:
                return str(match)
        return None

    def _first_value(self, record: dict[str, Any], candidates: tuple[str, ...]) -> str:
        key = self._find_key(record, candidates)
        if key is None:
            return ""
        value = record.get(key, "")
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _numeric_string(value: str) -> str:
        cleaned = re.sub(r"[^\d.\-]", "", value or "")
        if not cleaned:
            return ""
        try:
            return f"{float(cleaned):.2f}"
        except ValueError:
            return ""

    @staticmethod
    def _integer_string(value: str) -> str:
        cleaned = re.sub(r"[^\d\-]", "", value or "")
        if not cleaned:
            return ""
        try:
            return str(max(0, int(cleaned)))
        except ValueError:
            return ""

    @staticmethod
    def _extract_pdf_field(block: str, pattern: str) -> str:
        match = re.search(pattern, block, flags=re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip().splitlines()[0][:500]

    @staticmethod
    def _extract_currency(block: str) -> str:
        match = re.search(r"(?:price|amount|cost)?\s*[:\-]?\s*([$£€]?\s?\d+(?:[.,]\d{2})?)", block, flags=re.IGNORECASE)
        if not match:
            return ""
        value = re.sub(r"[^\d.]", "", match.group(1))
        if not value:
            return ""
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return ""

    @staticmethod
    def _extract_quantity(block: str) -> str:
        match = re.search(r"(?:qty|quantity|stock|units?)\s*[:\-]?\s*(\d+)", block, flags=re.IGNORECASE)
        return match.group(1) if match else ""

    @staticmethod
    def _missing_fields(*, title: str, price: str, quantity: str) -> list[str]:
        missing: list[str] = []
        if not title:
            missing.append("title")
        if not price:
            missing.append("price")
        if not quantity:
            missing.append("quantity")
        return missing

    @staticmethod
    def _confidence_for(
        *,
        title: str,
        sku: str,
        description: str,
        price: str,
        quantity: str,
        image_url: str,
    ) -> float:
        score = 0.0
        if title:
            score += 0.35
        if sku:
            score += 0.15
        if description:
            score += 0.15
        if price:
            score += 0.15
        if quantity:
            score += 0.10
        if image_url:
            score += 0.10
        return round(min(score, 0.99), 2)
