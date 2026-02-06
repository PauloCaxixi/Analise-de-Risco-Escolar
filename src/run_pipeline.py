import os
import pandas as pd
from preprocessing import load_raw_data, clean_data, make_columns_unique
from features import create_target, feature_engineering
from train import build_pipeline, split_data, save_model
from evaluate import evaluate_model


DATA_PATH = "data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
MODEL_PATH = "models/model.joblib"


def run_pipeline():
    print("\n🔹 Iniciando pipeline de Machine Learning...\n")

    # 1️⃣ Carregar dados
    print("📥 Carregando dados...")
    df = load_raw_data(DATA_PATH)

    # 2️⃣ Garantir colunas únicas
    print("🧹 Corrigindo colunas duplicadas...")
    df = make_columns_unique(df)

    # 3️⃣ Limpeza básica
    print("🧼 Limpando dados...")
    df = clean_data(df)

    # 4️⃣ Criar variável alvo
    print("🎯 Criando variável alvo...")
    df = create_target(df)

    # 5️⃣ Engenharia de features
    print("⚙️ Criando features...")
    df = feature_engineering(df)

    # 6️⃣ Separar treino e teste
    print("✂️ Separando treino e teste...")
    X_train, X_test, y_train, y_test = split_data(df)

    # 7️⃣ Construir pipeline
    print("🏗️ Construindo pipeline de ML...")
    pipeline = build_pipeline(X_train)

    # 8️⃣ Treinar modelo
    print("🤖 Treinando modelo...")
    pipeline.fit(X_train, y_train)

    # 9️⃣ Avaliar modelo
    print("📊 Avaliando modelo...")
    evaluate_model(pipeline, X_test, y_test)

    # 🔟 Salvar modelo
    print("💾 Salvando modelo...")
    save_model(pipeline, MODEL_PATH)

    print("\n✅ Pipeline finalizada com sucesso!")
    print(f"📁 Modelo salvo em: {MODEL_PATH}")


if __name__ == "__main__":
    run_pipeline()

