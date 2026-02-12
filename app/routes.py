from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from app.schemas.predict import PredictRequest, PredictResponse
from app.schemas.health import HealthResponse
from app.schemas.version import VersionResponse
from app.services.predictor import predict
from app.services.model_loader import load_metadata

router = APIRouter()

API_VERSION = "v1.0.0"


@router.get("/health", response_model=HealthResponse)
def health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/version", response_model=VersionResponse)
def version() -> Dict[str, Any]:
    md = load_metadata()
    return {
        "api_version": API_VERSION,
        "model_version": md.get("model_version", "unknown"),
        "test_year": md.get("test_year"),
        "metrics": md.get("metrics", {}),
        "dropped_all_null_columns": md.get("dropped_all_null_columns", []),
    }


@router.get("/features")
def features() -> Dict[str, List[str]]:
    md = load_metadata()
    feats = md.get("features", [])
    if not isinstance(feats, list):
        feats = []
    return {"features": feats}


@router.get("/example")
def example() -> Dict[str, Any]:
    # Exemplo mínimo: você pode mandar subset, o servidor completa o resto com null.
    return {
        "features": {
            "IDADE_ALUNO_2020": 12,
            "IAN_2022": 7.5,
            "IDA_2022": 6.0,
            "IEG_2022": 8.0,
        }
    }


@router.post("/predict", response_model=PredictResponse)
def predict_route(payload: PredictRequest) -> Dict[str, Any]:
    try:
        return predict(payload)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
