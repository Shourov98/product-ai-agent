from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.agents.restock_agent import RestockAgent
from app.config import get_settings
from app.schemas.forecasting import CSVUploadResponse, ForecastRequest, ForecastResponse, SalesRecord
from app.services.csv_parser import CSVParser
from app.services.forecasting_engine import ForecastingEngine


router = APIRouter(prefix="/forecast", tags=["forecasting"])


@router.post(
    "/demand",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
)
async def forecast_demand(payload: ForecastRequest) -> ForecastResponse:
    try:
        forecast = ForecastingEngine().forecast(payload)
        return RestockAgent().generate_recommendation(forecast)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/upload-csv",
    response_model=CSVUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_sales_csv(file: UploadFile = File(...)) -> CSVUploadResponse:
    content = await file.read()
    parser = CSVParser()
    parsed = parser.parse(content)
    settings = get_settings()
    if not settings.mongodb_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MONGODB_URI is required to store uploaded sales records.",
        )

    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="pymongo is required to store uploaded sales records.",
        ) from exc

    client = MongoClient(settings.mongodb_uri)
    try:
        collection = client[settings.mongodb_db_name]["sales_records"]
        if parsed.parsed_records:
            collection.insert_many([record.model_dump(mode="json") for record in parsed.parsed_records])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not store sales records: {exc}",
        ) from exc
    finally:
        client.close()
    return parsed


@router.get(
    "/demand/{product_id}",
    response_model=ForecastResponse,
    status_code=status.HTTP_200_OK,
)
async def forecast_product_demand(product_id: str) -> ForecastResponse:
    settings = get_settings()
    if not settings.mongodb_uri:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MONGODB_URI is required to load sales records.",
        )

    try:
        from pymongo import MongoClient
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="pymongo is required to load sales records.",
        ) from exc

    cutoff = date.today() - timedelta(days=90)
    client = MongoClient(settings.mongodb_uri)
    try:
        collection = client[settings.mongodb_db_name]["sales_records"]
        records = list(collection.find({"product_id": product_id, "date": {"$gte": cutoff.isoformat()}}))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not load sales records: {exc}",
        ) from exc
    finally:
        client.close()

    if not records:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product sales history not found.")

    sales_history = [_sales_record_from_mongo(record) for record in records]
    request = ForecastRequest(product_id=product_id, sales_history=sales_history, current_stock=0)
    try:
        forecast = ForecastingEngine().forecast(request)
        return RestockAgent().generate_recommendation(forecast)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


def _sales_record_from_mongo(payload: dict[str, Any]) -> SalesRecord:
    payload.pop("_id", None)
    return SalesRecord.model_validate(payload)
