import joblib
import pandas as pd
from src.config import MODEL_PATH


def test_model_prediction():
    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "ANO_PEDE": 2024,
        "INDE_2022": 6.0,
        "INDE_2023": 5.5,
        "INDE_2024": 5.0
    }])

    pred = model.predict(sample)
    assert pred[0] in [0, 1]
