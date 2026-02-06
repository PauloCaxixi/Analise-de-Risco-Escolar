import joblib
from src.config import MODEL_PATH
from sklearn.metrics import classification_report, accuracy_score, f1_score


def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo usando métricas de classificação
    """

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n📊 Resultados da Avaliação do Modelo:")
    print(f"Acurácia: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
