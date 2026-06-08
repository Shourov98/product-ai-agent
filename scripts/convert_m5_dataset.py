from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_DATASET_DIR = Path("Datasets") / "m5-forecasting-accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the M5 Forecasting dataset into product_id,date,units_sold rows."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory containing calendar.csv and sales_train_evaluation.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATASET_DIR / "converted_sales_records_last90_first100.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--last-days",
        type=int,
        default=90,
        help="Number of trailing M5 daily columns to export.",
    )
    parser.add_argument(
        "--limit-products",
        type=int,
        default=100,
        help="Maximum number of product-store series to export. Use 0 for all.",
    )
    parser.add_argument(
        "--sales-file",
        default="sales_train_evaluation.csv",
        help="Sales CSV filename inside dataset-dir.",
    )
    return parser.parse_args()


def load_calendar_dates(calendar_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with calendar_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            day_key = str(row.get("d", "")).strip()
            date_value = str(row.get("date", "")).strip()
            if day_key and date_value:
                mapping[day_key] = date_value
    return mapping


def day_number(day_column: str) -> int:
    return int(day_column.split("_", maxsplit=1)[1])


def select_day_columns(fieldnames: list[str], last_days: int) -> list[str]:
    day_columns = [field for field in fieldnames if field.startswith("d_")]
    ordered = sorted(day_columns, key=day_number)
    if last_days <= 0 or last_days >= len(ordered):
        return ordered
    return ordered[-last_days:]


def convert_m5_dataset(
    *,
    dataset_dir: Path,
    output_path: Path,
    sales_file: str,
    last_days: int,
    limit_products: int,
) -> tuple[int, int]:
    calendar_dates = load_calendar_dates(dataset_dir / "calendar.csv")
    sales_path = dataset_dir / sales_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported_records = 0
    exported_products = 0

    with sales_path.open("r", encoding="utf-8-sig", newline="") as sales_handle:
        reader = csv.DictReader(sales_handle)
        if reader.fieldnames is None:
            raise ValueError(f"{sales_path} has no header row.")

        day_columns = select_day_columns(reader.fieldnames, last_days)
        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=[
                    "product_id",
                    "date",
                    "units_sold",
                    "item_id",
                    "dept_id",
                    "cat_id",
                    "store_id",
                    "state_id",
                ],
            )
            writer.writeheader()

            for row in reader:
                if limit_products > 0 and exported_products >= limit_products:
                    break

                product_id = str(row.get("id", "")).strip()
                if not product_id:
                    continue

                exported_products += 1
                for day_column in day_columns:
                    date_value = calendar_dates.get(day_column)
                    if date_value is None:
                        continue
                    writer.writerow(
                        {
                            "product_id": product_id,
                            "date": date_value,
                            "units_sold": row.get(day_column, "0") or "0",
                            "item_id": row.get("item_id", ""),
                            "dept_id": row.get("dept_id", ""),
                            "cat_id": row.get("cat_id", ""),
                            "store_id": row.get("store_id", ""),
                            "state_id": row.get("state_id", ""),
                        }
                    )
                    exported_records += 1

    return exported_products, exported_records


def main() -> None:
    args = parse_args()
    products, records = convert_m5_dataset(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        sales_file=args.sales_file,
        last_days=args.last_days,
        limit_products=args.limit_products,
    )
    print(f"Exported {records} sales records for {products} products to {args.output}")


if __name__ == "__main__":
    main()
