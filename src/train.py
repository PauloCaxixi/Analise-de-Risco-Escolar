from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from src.features import coerce_numeric
from src.features import split_features
from src.features import standardize_columns
try:
    import joblib  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Dependência ausente: joblib. Instale em requirements.txt") from exc


LOGGER = logging.getLogger("pede-train")

# =========================
# DATASET BUILDING (t -> t+1)
# =========================
@dataclass(frozen=True)
class PairSpec:
    x_sheet: str
    y_sheet: str
    x_year: int
    y_year: int


PAIR_SPECS: List[PairSpec] = [
    PairSpec(x_sheet="PEDE2022", y_sheet="PEDE2023", x_year=2022, y_year=2023),
    PairSpec(x_sheet="PEDE2023", y_sheet="PEDE2024", x_year=2023, y_year=2024),
]


def _read_sheet(xlsx_path: Path, sheet: str) -> pd.DataFrame:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Arquivo XLSX não encontrado: {xlsx_path}")
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet, engine=None)
    except ValueError as exc:
        raise ValueError(f"Sheet inválida: {sheet}. Erro: {exc}") from exc
    if df.empty:
        raise ValueError(f"Sheet '{sheet}' está vazia.")
    return standardize_columns(df)


def _to_binary_target(series: pd.Series) -> pd.Series:
    """
    Converte Defasagem para 0/1 de forma robusta.
    Aceita: 0/1, True/False, "Sim/Não", "S/N", "x", etc.
    """
    if series.dtype.kind in {"i", "u"}:
        return (series.fillna(0).astype(int) > 0).astype(int)

    if series.dtype.kind == "b":
        return series.fillna(False).astype(bool).astype(int)

    s = series.astype(str).str.strip().str.casefold()
    positives = {"1", "true", "sim", "s", "yes", "y", "x"}
    negatives = {"0", "false", "nao", "não", "n", "no", ""}

    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out[s.isin(positives)] = 1.0
    out[s.isin(negatives)] = 0.0

    # fallback: tenta numérico
    num = pd.to_numeric(series, errors="coerce")
    out = out.fillna((num.fillna(0) > 0).astype(int).astype(float))
    return out.astype(int)


def _merge_pair(df_x: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
    if "RA" not in df_x.columns or "RA" not in df_y.columns:
        raise KeyError("Coluna obrigatória 'RA' ausente em uma das sheets.")

    x = df_x.copy()
    y = df_y.copy()

    x["RA"] = x["RA"].astype(str).str.strip()
    y["RA"] = y["RA"].astype(str).str.strip()

    # Target precisa existir no ano y
    if "Defasagem" not in y.columns:
        raise KeyError("Coluna obrigatória 'Defasagem' ausente na sheet de target.")

    # Merge (inner): só alunos presentes nos dois anos
    merged = x.merge(
        y[["RA", "Defasagem"]],
        on="RA",
        how="inner",
        suffixes=("", "_y"),
    )
    merged = merged.rename(columns={"Defasagem_y": "Defasagem_next_year"})
    return merged


def build_longitudinal_dataset(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna (train_df, test_df) usando split temporal:
      train: 2022->2023
      test : 2023->2024
    Se o par test ficar vazio, faz split aleatório depois (handled no train()).
    """
    train_pair = PAIR_SPECS[0]
    test_pair = PAIR_SPECS[1]

    df_22 = _read_sheet(xlsx_path, train_pair.x_sheet)
    df_23 = _read_sheet(xlsx_path, train_pair.y_sheet)
    train_df = _merge_pair(df_22, df_23)

    df_23x = _read_sheet(xlsx_path, test_pair.x_sheet)
    df_24 = _read_sheet(xlsx_path, test_pair.y_sheet)
    test_df = _merge_pair(df_23x, df_24)

    return train_df, test_df


# =========================
# FEATURES
# =========================
def choose_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Mantém compatibilidade com o restante do código,
    mas delega a seleção para src/features.py
    """
    return split_features(df)

def make_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def class_weights_from_y(y: np.ndarray) -> np.ndarray:
    """
    Peso inverso à frequência para lidar com desbalanceamento.
    Retorna sample_weight por linha.
    """
    y_int = y.astype(int)
    n = len(y_int)
    pos = int((y_int == 1).sum())
    neg = n - pos
    if pos == 0 or neg == 0:
        return np.ones(n, dtype="float64")
    w_pos = n / (2.0 * pos)
    w_neg = n / (2.0 * neg)
    return np.where(y_int == 1, w_pos, w_neg).astype("float64")


# =========================
# TRAIN / EVAL / SAVE
# =========================
def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[Pipeline, Dict[str, Any], List[str]]:
    if "Defasagem_next_year" not in train_df.columns:
        raise KeyError("Coluna alvo 'Defasagem_next_year' ausente no dataset de treino.")

    y_train = _to_binary_target(train_df["Defasagem_next_year"]).to_numpy()
    y_test: Optional[np.ndarray] = None

    cat_cols, num_cols = choose_feature_columns(train_df)
    feature_cols = cat_cols + num_cols

    x_train = train_df[feature_cols].copy()

    if not test_df.empty and "Defasagem_next_year" in test_df.columns:
        y_test = _to_binary_target(test_df["Defasagem_next_year"]).to_numpy()
        x_test = test_df[feature_cols].copy()
    else:
        x_test = None

    pre = make_preprocessor(cat_cols, num_cols)

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=4,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", pre),
            ("model", model),
        ]
    )

    sw = class_weights_from_y(y_train)
    pipe.fit(x_train, y_train, model__sample_weight=sw)

    metrics: Dict[str, Any] = {"n_train": int(len(train_df)), "n_test": int(len(test_df))}
    # Avaliação
    if x_test is not None and y_test is not None and len(y_test) > 0 and len(np.unique(y_test)) > 1:
        proba = pipe.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        metrics.update(
            {
                "roc_auc": float(roc_auc_score(y_test, proba)),
                "pr_auc": float(average_precision_score(y_test, proba)),
                "f1": float(f1_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
            }
        )
    else:
        # Sem avaliação temporal consistente: devolve somente contagens
        metrics.update(
            {
                "roc_auc": None,
                "pr_auc": None,
                "f1": None,
                "precision": None,
                "recall": None,
            }
        )

    return pipe, metrics, feature_cols


def save_artifacts(
    pipeline: Pipeline,
    feature_cols: List[str],
    metrics: Dict[str, Any],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    joblib.dump(pre, out_dir / "preprocessor.joblib")
    joblib.dump(model, out_dir / "model.joblib")

    metadata = {
        "project": "Datathon Passos Magicos - PEDE",
        "model_name": "hgb_classifier",
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "task": "binary_classification_proba_to_risk_label",
        "target": "Defasagem(t+1) -> P(defasagem)",
        "risk_thresholds": {
            "muito_alto_gte": 0.85,
            "alto_gte": 0.70,
            "medio_gte": 0.50,
        },
        "feature_columns": feature_cols,
        "metrics": metrics,
        "notes": [
            "Pipeline temporal: train 2022->2023, test 2023->2024 (quando disponível).",
            "A saída do modelo é probabilidade; o dashboard converte para {Muito Alto, Alto, Médio, Regular} por thresholds.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Treino longitudinal PEDE (t -> t+1).")
    p.add_argument(
        "--xlsx",
        type=str,
        required=True,
        help="Caminho do XLSX (BASE DE DADOS PEDE 2024 - DATATHON.xlsx).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(Path("app") / "model"),
        help="Diretório de saída dos artefatos (model.joblib, preprocessor.joblib, metadata.json).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Nível de log.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    LOGGER.info("Lendo e montando dataset longitudinal...")
    train_df, test_df = build_longitudinal_dataset(xlsx_path)

    # Converte numéricos de forma centralizada (mesma regra do app)
    train_df = coerce_numeric(train_df)
    test_df = coerce_numeric(test_df)

    LOGGER.info("Treinando modelo...")
    pipe, metrics, feature_cols = train_and_evaluate(train_df, test_df)

    LOGGER.info("Salvando artefatos em %s", out_dir)
    save_artifacts(pipe, feature_cols, metrics, out_dir)

    LOGGER.info("OK. Métricas: %s", metrics)


if __name__ == "__main__":
    main()