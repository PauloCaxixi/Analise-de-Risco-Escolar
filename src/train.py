from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
from src.transformers import ToNumeric, ToString


import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# Paths
# =========================

@dataclass(frozen=True)
class TrainPaths:
    project_root: Path
    excel_path: Path
    model_dir: Path

    @staticmethod
    def resolve() -> "TrainPaths":
        # Suporta execução via `python -m src.train` e também Jupyter (sem __file__)
        try:
            project_root = Path(__file__).resolve().parents[1]
        except NameError:
            project_root = Path.cwd().resolve()

        excel_path = project_root / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
        model_dir = project_root / "app" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        return TrainPaths(project_root=project_root, excel_path=excel_path, model_dir=model_dir)


# =========================
# Leitura / preparo
# =========================

def _read_sheet(excel_path: Path, sheet: str, year: int) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel não encontrado: {excel_path}")

    try:
        df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl").copy()
    except ImportError as e:
        raise ImportError("Dependência ausente para ler .xlsx. Instale: pip install openpyxl") from e

    df.columns = [str(c).strip() for c in df.columns]

    # Normaliza nomes tolerando variações simples (case/espacos)
    col_map = {c.lower(): c for c in df.columns}
    ra_col = col_map.get("ra")
    defas_col = col_map.get("defasagem")

    if not ra_col:
        raise ValueError(f"Sheet {sheet} sem coluna RA (colunas: {list(df.columns)})")
    if not defas_col:
        raise ValueError(f"Sheet {sheet} sem coluna Defasagem (colunas: {list(df.columns)})")

    if ra_col != "RA":
        df = df.rename(columns={ra_col: "RA"})
    if defas_col != "Defasagem":
        df = df.rename(columns={defas_col: "Defasagem"})

    df["RA"] = df["RA"].astype(str).str.strip()
    df["ref_year"] = int(year)

    # Padroniza NA sem FutureWarning de downcast
    df = df.where(pd.notna(df), np.nan)
    return df


def _to_float_series(s: pd.Series) -> pd.Series:
    # Aceita "1,23" e tenta limpar ruídos comuns
    s2 = (
        s.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)   # separador de milhar comum
        .str.replace(",", ".", regex=False)  # decimal PT-BR
    )
    out = pd.to_numeric(s2, errors="coerce")
    return out.astype(float)


def _build_longitudinal_pairs(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Dataset temporal:
      X_t  = features do ano t
      y_t1 = Defasagem do ano t+1
    Treino: 2022->2023
    Teste:  2023->2024
    """
    df22 = df_all[df_all["ref_year"] == 2022].copy()
    df23 = df_all[df_all["ref_year"] == 2023].copy()
    df24 = df_all[df_all["ref_year"] == 2024].copy()

    train_pairs = df22.merge(
        df23[["RA", "Defasagem"]].rename(columns={"Defasagem": "Defasagem_t1"}),
        on="RA",
        how="inner",
    )

    test_pairs = df23.merge(
        df24[["RA", "Defasagem"]].rename(columns={"Defasagem": "Defasagem_t1"}),
        on="RA",
        how="inner",
    )

    drop_cols = {"Defasagem_t1", "Nome", "Nome Anonimizado"}
    feature_cols = [c for c in train_pairs.columns if c not in drop_cols]

    X_train = train_pairs[feature_cols].copy()
    X_test = test_pairs[feature_cols].copy()

    y_train = _to_float_series(train_pairs["Defasagem_t1"]).copy()
    y_test = _to_float_series(test_pairs["Defasagem_t1"]).copy()

    if y_train.isna().all():
        raise ValueError("y_train ficou todo NaN após conversão. Verifique Defasagem/Defasagem_t1 no PEDE2023.")
    if y_test.isna().all():
        raise ValueError("y_test ficou todo NaN após conversão. Verifique Defasagem/Defasagem_t1 no PEDE2024.")

    # Padroniza NA sem FutureWarning de downcast
    X_train = X_train.where(pd.notna(X_train), np.nan)
    X_test = X_test.where(pd.notna(X_test), np.nan)

    return X_train, y_train, X_test, y_test


def _split_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separação robusta com base no conteúdo do TREINO:
    - numérica: 100% dos valores não-nulos convertem para número
    - categórica: caso contrário
    """
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in X.columns:
        if col in ("RA", "ref_year"):
            categorical_cols.append(col)
            continue

        s = X[col].dropna()
        if s.empty:
            categorical_cols.append(col)
            continue

        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().all():
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _build_ohe() -> OneHotEncoder:
    # Compatível com versões novas e antigas do scikit-learn
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# =========================
# Train
# =========================

def train() -> None:
    paths = TrainPaths.resolve()

    df_all = pd.concat(
        [
            _read_sheet(paths.excel_path, "PEDE2022", 2022),
            _read_sheet(paths.excel_path, "PEDE2023", 2023),
            _read_sheet(paths.excel_path, "PEDE2024", 2024),
        ],
        ignore_index=True,
    )

    X_train, y_train, X_test, y_test = _build_longitudinal_pairs(df_all)

    # Remove colunas 100% nulas no treino
    all_null_cols = [c for c in X_train.columns if X_train[c].isna().all()]
    if all_null_cols:
        X_train = X_train.drop(columns=all_null_cols)
        X_test = X_test.drop(columns=[c for c in all_null_cols if c in X_test.columns])

    num_cols, cat_cols = _split_columns(X_train)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("to_numeric", ToNumeric()),
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("to_string", ToString()),
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", _build_ohe()),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        random_state=42,
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    pred_test = pipeline.predict(X_test)
    pred_test_clipped = np.clip(pred_test, -5, 3)

    mae = float(mean_absolute_error(y_test, pred_test_clipped))
    # FIX: sklearn atual no seu ambiente não aceita `squared=...`
    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test_clipped)))

    # Persistência (artefatos separados)
    model_path = paths.model_dir / "model.joblib"
    preproc_path = paths.model_dir / "preprocessor.joblib"
    metadata_path = paths.model_dir / "metadata.json"

    joblib.dump(pipeline.named_steps["model"], model_path)
    joblib.dump(pipeline.named_steps["preprocessor"], preproc_path)

    if not model_path.exists():
        raise RuntimeError(f"Falha ao salvar model.joblib em: {model_path}")
    if not preproc_path.exists():
        raise RuntimeError(f"Falha ao salvar preprocessor.joblib em: {preproc_path}")

    metadata: Dict[str, Any] = {
        "model_version": "v2.0.0-next-year",
        "trained_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "task": "regression_next_year_defasagem",
        "target": "Defasagem(t+1)",
        "train_pairs": ["2022->2023"],
        "test_pairs": ["2023->2024"],
        "metrics": {"mae": mae, "rmse": rmse},
        "features": list(X_train.columns),
        "numeric_cols": num_cols,
        "categorical_cols": cat_cols,
        "notes": (
            "Modelo longitudinal: usa snapshot do ano t para prever Defasagem no ano t+1. "
            "Transformer numérico força coerção; strings inesperadas viram NaN e são imputadas."
        ),
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Treino longitudinal concluído.")
    print({"mae": mae, "rmse": rmse})
    print(f"OK: model.joblib -> {model_path}")
    print(f"OK: preprocessor.joblib -> {preproc_path}")
    print(f"OK: metadata.json -> {metadata_path}")


if __name__ == "__main__":
    train()
