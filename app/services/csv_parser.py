from __future__ import annotations

import csv
import logging
from datetime import date
from io import StringIO

from app.schemas.forecasting import CSVUploadResponse, SalesRecord


logger = logging.getLogger(__name__)


class CSVParser:
    DATE_COLUMNS = ("date", "Date", "order_date", "ds")
    UNITS_COLUMNS = ("units_sold", "sales", "quantity", "qty", "y")
    PRODUCT_COLUMNS = ("product_id", "item_id", "sku", "SKU", "product")

    def parse(self, payload: bytes) -> CSVUploadResponse:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(StringIO(text))
        if reader.fieldnames is None:
            return CSVUploadResponse(parsed_records=[], product_ids_found=[], row_count=0)

        date_column = self._find_column(reader.fieldnames, self.DATE_COLUMNS)
        units_column = self._find_column(reader.fieldnames, self.UNITS_COLUMNS)
        product_column = self._find_column(reader.fieldnames, self.PRODUCT_COLUMNS)

        if date_column is None or units_column is None or product_column is None:
            logger.warning("CSV is missing required forecasting columns.")
            return CSVUploadResponse(parsed_records=[], product_ids_found=[], row_count=0)

        parsed_records: list[SalesRecord] = []
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            try:
                parsed_records.append(
                    SalesRecord(
                        date=self._parse_date(row.get(date_column, "")),
                        units_sold=self._parse_units(row.get(units_column, "")),
                        product_id=str(row.get(product_column, "")).strip(),
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed CSV row %s: %s", row_count, exc)

        product_ids = sorted({record.product_id for record in parsed_records})
        return CSVUploadResponse(
            parsed_records=parsed_records,
            product_ids_found=product_ids,
            row_count=row_count,
        )

    @staticmethod
    def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
        for candidate in candidates:
            if candidate in fieldnames:
                return candidate
        lowered = {field.lower(): field for field in fieldnames}
        for candidate in candidates:
            match = lowered.get(candidate.lower())
            if match is not None:
                return match
        return None

    @staticmethod
    def _parse_date(value: str) -> date:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("date is empty")
        try:
            return date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"invalid date {cleaned!r}; expected YYYY-MM-DD") from exc

    @staticmethod
    def _parse_units(value: str) -> int:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("units_sold is empty")
        units = int(float(cleaned))
        if units < 0:
            raise ValueError("units_sold cannot be negative")
        return units
