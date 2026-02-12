import json
import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    confusion_matrix,
)

from src.preprocessing import preprocess_all
from src.feature_engineering import split_features_target
from src.config import MODEL_PATH, TEST_YEAR


def evaluate() -> None:
    print("Carregando dados...")
    df = preprocess_all()

    print(f"Filtrando dados de teste para o ano {TEST_YEAR}...")
    test_df = df[df["ANO_REFERENCIA"] == TEST_YEAR]

    if test_df.empty:
        raise ValueError(f"Nenhum dado encontrado para TEST_YEAR={TEST_YEAR}")

    X_test, y_test = split_features_target(test_df)

    print("Carregando modelo treinado...")
    pipeline = joblib.load(MODEL_PATH / "model.joblib")

    print("Gerando previsões...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    if y_test.nunique() < 2:
        raise ValueError("O conjunto de teste precisa ter pelo menos duas classes para calcular ROC AUC.")

    print("Calculando métricas...")
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "n_samples": len(y_test),
        "positive_rate": float(y_test.mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    output_path = MODEL_PATH / "evaluation.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Avaliação final salva em {output_path}")


if __name__ == "__main__":
    evaluate()
