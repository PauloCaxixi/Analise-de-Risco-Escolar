from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# COLUMN NORMALIZATION
# =========================
RENAME_MAP: Dict[str, str] = {
    # Identificação
    "Nome Anonimizado": "Nome",
    "Escola": "Instituição de ensino",

    # Disciplinas
    "Mat": "Matem",
    "Por": "Portug",
    "Ing": "Inglês",

    # Demografia
    "Data de Nasc": "Ano nasc",
    "Idade": "Idade 22",

    # Flags
    "Fase Ideal": "Fase ideal",
}


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza nomes de colunas entre PEDE2022 / PEDE2023 / PEDE2024.
    Esta função DEVE ser usada tanto no treino quanto no app (inferência).
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.rename(columns={k: v for k, v in RENAME_MAP.items() if k in out.columns})
    return out


# =========================
# FEATURE SELECTION
# =========================
ALLOWED_FEATURES: set[str] = {
    # Contexto
    "Fase",
    "Turma",
    "Gênero",
    "Ano ingresso",
    "Instituição de ensino",

    # Demografia
    "Idade 22",

    # Histórico pedagógico
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "INDE 22",

    # Indicadores
    "IAA",
    "IEG",
    "IPS",
    "IDA",
    "IPV",
    "IAN",

    # Engajamento
    "Cg",
    "Cf",
    "Ct",
    "Nº Av",

    # Disciplinas
    "Matem",
    "Portug",
    "Inglês",

    # Flags auxiliares (não leakage direto)
    "Atingiu PV",
    "Indicado",
    "Fase ideal",
    "Destaque IEG",
    "Destaque IDA",
    "Destaque IPV",

    # Apoio
    "Rec Psicologia",
    "Rec Av1",
    "Rec Av2",
    "Rec Av3",
    "Rec Av4",

    # Avaliadores
    "Avaliador1",
    "Avaliador2",
    "Avaliador3",
    "Avaliador4",
}


FORBIDDEN_FEATURES: set[str] = {
    # Identificadores
    "RA",
    "Nome",

    # Targets / leakage
    "Defasagem",
    "Defasagem_next_year",
}


CATEGORICAL_FEATURES: set[str] = {
    "Fase",
    "Turma",
    "Gênero",
    "Instituição de ensino",
}


def split_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Retorna (categorical_cols, numeric_cols) já filtradas e prontas
    para uso no ColumnTransformer.
    """
    cols = [
        c
        for c in df.columns
        if c in ALLOWED_FEATURES and c not in FORBIDDEN_FEATURES
    ]

    if not cols:
        raise ValueError("Nenhuma feature válida encontrada após filtros.")

    cat_cols = [c for c in cols if c in CATEGORICAL_FEATURES]
    num_cols = [c for c in cols if c not in CATEGORICAL_FEATURES]

    return cat_cols, num_cols


# =========================
# NUMERIC COERCION
# =========================
NUMERIC_COLUMNS: set[str] = {
    "Idade 22",
    "Ano ingresso",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "INDE 22",
    "IAA",
    "IEG",
    "IPS",
    "IDA",
    "IPV",
    "IAN",
    "Cg",
    "Cf",
    "Ct",
    "Nº Av",
    "Matem",
    "Portug",
    "Inglês",
}


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas numéricas conhecidas para float,
    ignorando erros (NaN quando inválido).
    """
    out = df.copy()
    for c in NUMERIC_COLUMNS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =========================
# UTILITIES
# =========================
def calc_media_disciplinas(row: pd.Series) -> float:
    """
    Calcula média simples entre Matem, Portug e Inglês,
    ignorando valores ausentes.
    """
    vals: List[float] = []
    for c in ("Matem", "Portug", "Inglês"):
        if c in row.index:
            v = pd.to_numeric(row.get(c), errors="coerce")
            if not pd.isna(v):
                vals.append(float(v))
    if not vals:
        return float("nan")
    return float(np.mean(vals))