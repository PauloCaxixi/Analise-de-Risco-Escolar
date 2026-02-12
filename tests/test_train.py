from pathlib import Path
from src.train import train
from src.config import MODEL_PATH


def test_train_creates_model_file():
    train()
    assert (MODEL_PATH / "model.joblib").exists()
