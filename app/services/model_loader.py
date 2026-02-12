from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib

MODEL_DIR = Path(__file__).resolve().parents[2] / "app/model"


@lru_cache
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")


@lru_cache
def load_metadata() -> Dict[str, Any]:
    path = MODEL_DIR / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata.json não encontrado em {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
