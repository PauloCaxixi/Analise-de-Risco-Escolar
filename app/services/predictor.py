from __future__ import annotations

from typing import Any, Dict, TypedDict

import joblib
import numpy as np
import pandas as pd

from app.schemas.predict import PredictRequest
from app.services.model_loader import load_metadata
from app.services.validation import validate_features


class PredictionOut(TypedDict):
    model_version: str
    predicted_defasagem_next_year: float
    predicted_defasagem_rounded: int
    risk_level: str
    status: str
    recommended_action: str
    feature_coverage: float
    features_filled: int
    features_expected: int


def _normalize_key(key: str) -> str:
    return " ".join(key.strip().split()).casefold()


def _build_row(features: Dict[str, Any], expected_features: list[str]) -> Dict[str, Any]:
    incoming_map = {_normalize_key(k): k for k in features.keys()}
    row: Dict[str, Any] = {k: None for k in expected_features}

    for feat in expected_features:
        nk = _normalize_key(feat)
        if nk in incoming_map:
            original_key = incoming_map[nk]
            row[feat] = features.get(original_key)

    return row


def _interpret_defasagem_next_year(defasagem_next: int) -> Dict[str, str]:
    """
    Interpretação pedagógica: o que fazer baseado na Defasagem prevista (t+1).
    """
    if defasagem_next <= -2:
        return {
            "risk_level": "CRÍTICO",
            "status": "Risco crítico de defasagem no próximo ano",
            "recommended_action": "Plano intensivo: reforço + acompanhamento semanal + intervenção psicopedagógica.",
        }
    if defasagem_next == -1:
        return {
            "risk_level": "ALTO",
            "status": "Alto risco de defasagem no próximo ano",
            "recommended_action": "Acompanhamento contínuo e reforço direcionado nas disciplinas de base (Mat/Por/Ing).",
        }
    if defasagem_next == 0:
        return {
            "risk_level": "MÉDIO",
            "status": "Em risco (pode entrar em defasagem no próximo ano)",
            "recommended_action": "Monitoramento mensal, metas de engajamento e revisão de conteúdos-chave.",
        }
    return {
        "risk_level": "BAIXO",
        "status": "Baixo risco de defasagem no próximo ano",
        "recommended_action": "Manter acompanhamento regular e estímulo a metas de engajamento (IEG).",
    }


def _load_artifacts() -> tuple[Any, Any, str, list[str]]:
    metadata = load_metadata()
    expected_features = metadata.get("features")
    model_version = str(metadata.get("model_version", "unknown"))

    if not isinstance(expected_features, list) or not expected_features:
        raise ValueError("metadata.json inválido: lista de features ausente")

    # Artefatos salvos em app/model/...
    # Reaproveita o padrão já correto do seu model_loader (path robusto) via metadata,
    # aqui carregamos direto pelo caminho relativo ao projeto para evitar circular import.
    # (Se preferir, eu adapto para usar seu load_model/load_preprocessor.)
    from pathlib import Path
    model_dir = Path(__file__).resolve().parents[1] / "model"
    model = joblib.load(model_dir / "model.joblib")
    preproc = joblib.load(model_dir / "preprocessor.joblib")

    return model, preproc, model_version, expected_features


def predict(payload: PredictRequest) -> PredictionOut:
    validate_features(payload.features)

    model, preproc, model_version, expected_features = _load_artifacts()

    row = _build_row(payload.features, expected_features)
    df = pd.DataFrame([row], columns=expected_features)

    filled = sum(1 for v in row.values() if v is not None)
    expected = len(expected_features)
    coverage = filled / expected if expected else 0.0

    X = preproc.transform(df)
    y_hat = float(model.predict(X)[0])

    # Clipa pro domínio real do PEDE
    y_hat = float(np.clip(y_hat, -5, 3))
    y_round = int(np.clip(int(np.rint(y_hat)), -5, 3))

    interp = _interpret_defasagem_next_year(y_round)

    return {
        "model_version": model_version,
        "predicted_defasagem_next_year": round(y_hat, 3),
        "predicted_defasagem_rounded": y_round,
        "risk_level": interp["risk_level"],
        "status": interp["status"],
        "recommended_action": interp["recommended_action"],
        "feature_coverage": round(coverage, 3),
        "features_filled": filled,
        "features_expected": expected,
    }
