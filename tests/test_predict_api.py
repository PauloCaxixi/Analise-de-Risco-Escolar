from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_endpoint():
    payload = {
        "features": {
            "IDADE_ALUNO_2020": 12,
            "IAN_2022": 7.5,
            "IDA_2022": 6.0,
            "IEG_2022": 8.0,
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "risk_score" in body
    assert "risk_label" in body
