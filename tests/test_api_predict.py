from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest


class _DummyPreprocessor:
    def transform(self, x):  # type: ignore[no-untyped-def]
        # Retorna matriz numérica simples
        return np.zeros((len(x), 3), dtype="float64")


class _DummyModel:
    def predict_proba(self, x):  # type: ignore[no-untyped-def]
        # Score fixo 0.82 => "Alto"
        n = x.shape[0]
        p = np.full((n, 2), 0.0, dtype="float64")
        p[:, 0] = 0.18
        p[:, 1] = 0.82
        return p


def test_predict_fallback_with_ra_only(client) -> None:
    resp = client.post("/predict?sheet=PEDE2022", json={"RA": "111"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert data["predictions"][0]["ra"] == "111"
    assert "risk_score" in data["predictions"][0]
    assert "risk_label" in data["predictions"][0]


def test_predict_batch(client) -> None:
    payload: List[Dict[str, Any]] = [{"RA": "111"}, {"RA": "222"}]
    resp = client.post("/predict?sheet=PEDE2022", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) == 2


def test_predict_invalid_json(client) -> None:
    resp = client.post("/predict", data="not-json", content_type="application/json")
    assert resp.status_code == 400


def test_predict_uses_model_when_bundle_available(client, monkeypatch: pytest.MonkeyPatch) -> None:
    import dashboard.app as dashboard_app

    def _fake_bundle():
        pre = _DummyPreprocessor()
        model = _DummyModel()
        metadata = {"feature_columns": ["INDE 22", "IEG", "IPS"], "model_version": "1.0.0"}
        return pre, model, metadata

    monkeypatch.setattr(dashboard_app, "_load_model_bundle", _fake_bundle)

    resp = client.post("/predict?sheet=PEDE2022", json={"RA": "111"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["used_model"] is True
    assert data["model_version"] == "1.0.0"
    assert data["predictions"][0]["risk_label"] in {"Alto", "Muito Alto", "Médio", "Regular"}