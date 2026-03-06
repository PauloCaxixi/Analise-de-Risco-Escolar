from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
import pytest

# Mock do Preprocessador para simular o comportamento do Scikit-Learn
class _DummyPreprocessor:
    def transform(self, x):
        # Retorna uma matriz de zeros simulando 3 features transformadas
        return np.zeros((len(x), 3), dtype="float64")

# Mock do Modelo para simular predição de probabilidade
class _DummyModel:
    def predict_proba(self, x):
        # Gera um score fixo de 0.82 (Classe 1), que deve resultar em "Alto" ou "Muito Alto"
        n = x.shape[0]
        p = np.full((n, 2), 0.0, dtype="float64")
        p[:, 0] = 0.18 # Probabilidade classe 0
        p[:, 1] = 0.82 # Probabilidade classe 1
        return p

def test_predict_fallback_with_ra_only(client) -> None:
    """Testa se o fallback funciona enviando apenas o RA."""
    resp = client.post("/predict?sheet=PEDE2022", json={"RA": "111"})
    assert resp.status_code == 200
    data = resp.get_json()
    
    assert "predictions" in data
    pred = data["predictions"][0]
    # Verifica se o RA retornado é o mesmo (independente de maiúscula/minúscula no JSON)
    assert str(pred.get("ra") or pred.get("RA")) == "111"
    assert "risk_score" in pred
    assert "risk_label" in pred

def test_predict_batch(client) -> None:
    """Testa predição enviando múltiplos alunos (Batch)."""
    payload: List[Dict[str, Any]] = [{"RA": "111"}, {"RA": "222"}]
    resp = client.post("/predict?sheet=PEDE2022", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["predictions"]) == 2

def test_predict_invalid_json(client) -> None:
    """Testa erro 400 ao enviar JSON malformado."""
    resp = client.post("/predict", data="not-a-json-string", content_type="application/json")
    assert resp.status_code == 400

def test_predict_uses_model_when_bundle_available(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Testa se a API usa o modelo Mockado em vez do fallback."""
    import dashboard.app as dashboard_app

    def _fake_bundle():
        pre = _DummyPreprocessor()
        model = _DummyModel()
        metadata = {
            "feature_columns": ["INDE 22", "IEG", "IPS"], 
            "model_version": "1.0.0-test"
        }
        return pre, model, metadata

    # Substitui a função real de carregar o modelo pela nossa fake
    monkeypatch.setattr(dashboard_app, "_load_model_bundle", _fake_bundle)

    resp = client.post("/predict?sheet=PEDE2022", json={"RA": "111"})
    assert resp.status_code == 200
    data = resp.get_json()
    
    # Verifica se os metadados do modelo mockado foram aplicados
    assert data.get("used_model") is True
    assert data.get("model_version") == "1.0.0-test"
    
    # O score 0.82 do DummyModel deve cair em um desses labels de risco
    assert data["predictions"][0]["risk_label"] in {"Alto", "Muito Alto"}

def test_predict_error_handling(client, monkeypatch: pytest.MonkeyPatch) -> None:
    """Testa se a API sobrevive a um erro interno de predição (Graceful Degradation)."""
    import dashboard.app as dashboard_app

    def _error_bundle():
        # Simula um bundle que existe mas falha ao processar
        class FailingModel:
            def predict_proba(self, x): raise ValueError("Simulated Model Crash")
        return _DummyPreprocessor(), FailingModel(), {"feature_columns": []}

    monkeypatch.setattr(dashboard_app, "_load_model_bundle", _error_bundle)

    # A rota deve retornar 200 usando fallback em vez de 500
    resp = client.post("/predict?sheet=PEDE2022", json={"RA": "111"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "predictions" in data