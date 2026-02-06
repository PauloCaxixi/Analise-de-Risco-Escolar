import pandas as pd
from src.config import INDE_RISK_THRESHOLD
import numpy as np


def create_target(df):
    # Seleciona colunas que começam com INDE
    inde_cols = [col for col in df.columns if col.startswith("INDE")]

    # Converte todas para numérico (erro vira NaN)
    for col in inde_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calcula o maior INDE disponível por aluno
    df["INDE_FINAL"] = df[inde_cols].max(axis=1)

    # Define risco de defasagem (regra de negócio)
    df["RISCO_DEFASAGEM"] = np.where(df["INDE_FINAL"] < 6, 1, 0)

    return df

import pandas as pd


def _find_column(df, possible_names):
    """
    Procura no dataframe uma coluna que bata com algum dos nomes possíveis
    """
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None


def feature_engineering(df):
    """
    Cria variáveis derivadas usadas no modelo
    """

    # 🔎 Encontrando colunas corretas automaticamente
    col_inde_2022 = _find_column(df, ["INDE 2022", "INDE 22"])
    col_inde_2023 = _find_column(df, ["INDE 2023", "INDE 23"])
    col_inde_2024 = _find_column(df, ["INDE 2024", "INDE 24"])
    col_ativo     = _find_column(df, ["Ativo", "Inativo"])

    # Converter para número (caso venha como texto)
    for col in [col_inde_2022, col_inde_2023, col_inde_2024]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 🎯 Média do INDE
    df["MEDIA_INDE"] = df[[col_inde_2022, col_inde_2023, col_inde_2024]].mean(axis=1)

    # 📉 Quantidade de anos com INDE < 6
    df["QTDE_INDE_BAIXO_6"] = (
        (df[[col_inde_2022, col_inde_2023, col_inde_2024]] < 6).sum(axis=1)
    )

    # 🔁 Padronizando aluno ativo
    if col_ativo:
        df["Ativo/ Inativo"] = df[col_ativo].apply(
            lambda x: 1 if str(x).strip().lower() in ["1", "sim", "ativo"] else 0
        )

    return df




def select_features(df: pd.DataFrame):
    """
    Separa X (features) e y (target)
    """
    X = df.drop(columns=["RISCO_DEFASAGEM", "INDE_FINAL"], errors="ignore")
    y = df["RISCO_DEFASAGEM"]

    return X, y
