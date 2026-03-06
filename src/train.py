from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

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
from pathlib import Path
import sys

# Caminho da pasta "src" correta do projeto
CURRENT_DIR = Path(__file__).resolve().parent      # .../Analise-de-Risco-Escolar/src
PROJECT_DIR = CURRENT_DIR.parent                   # .../Analise-de-Risco-Escolar

# Garante que o src correto esteja no sys.path
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Agora os imports SEMPRE virão do projeto atual
from src.features import standardize_columns, split_features, coerce_numeric
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

# NOVO: dataset multianual consolidado
MULTI_YEARS = ["PEDE2022", "PEDE2023", "PEDE2024"]

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


def build_longitudinal_dataset(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    NOVA VERSÃO:
    Monta dataset multianual real, unindo 2022 + 2023 + 2024,
    e cria target baseado na evolução histórica.
    """

    dfs = []
    for year, sheet in zip([2022, 2023, 2024], MULTI_YEARS):
        df = _read_sheet(xlsx_path, sheet)

        # 🔒 Remove duplicação de colunas
        df = df.loc[:, ~df.columns.duplicated()]

        # 🔒 Garante que não existe MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            df.index = df.index.to_flat_index()

        # 🔒 Reseta índice sempre
        df = df.reset_index(drop=True)

        # 🔒 Remove linhas com índice duplicado após reset
        df = df.copy()

        # 🔒 Adiciona ano
        df["ano"] = year

        dfs.append(df)

    # 🔥 Concatenar de forma totalmente segura
    full = pd.concat(dfs, ignore_index=True)
    full = full.loc[:, ~full.columns.duplicated()].reset_index(drop=True)

    
    # Criar target: aluno com alta defasagem / queda histórica
    full["Defasagem_next_year"] = (
        (pd.to_numeric(full.get("INDE 22"), errors="coerce") < 4)
        | (pd.to_numeric(full.get("IPS"), errors="coerce") < 0.4)
        | (pd.to_numeric(full.get("Pedra 22"), errors="coerce") <= 1)
    ).astype(int)

    # Separação temporal: treino = até 2023, teste = 2024
    train_df = full[full["ano"] < 2024].copy()
    test_df = full[full["ano"] == 2024].copy()

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

    # GARANTE QUE NUMÉRICAS SÃO NÚMEROS (Resolve o erro do 'Não')
    for c in num_cols:
        train_df[c] = pd.to_numeric(train_df[c], errors='coerce')
        if not test_df.empty:
            test_df[c] = pd.to_numeric(test_df[c], errors='coerce')

    # Corrige tipos mistos nas categóricas
    for c in cat_cols:
        train_df[c] = train_df[c].astype(str).replace("nan", "")
        if not test_df.empty:
            test_df[c] = test_df[c].astype(str).replace("nan", "")

    feature_cols = cat_cols + num_cols

    x_train_full = train_df[feature_cols].copy()

    if not test_df.empty and "Defasagem_next_year" in test_df.columns:
        y_test = _to_binary_target(test_df["Defasagem_next_year"]).to_numpy()

        test_aligned = test_df.copy()
        missing = [c for c in feature_cols if c not in test_aligned.columns]
        for c in missing:
            test_aligned[c] = np.nan

        x_train = x_train_full
        x_test = test_aligned[feature_cols].copy()
    else:
        # fallback: split estratificado no próprio train_df
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_train, idx_test = next(splitter.split(x_train_full, y_train))

        x_train = x_train_full.iloc[idx_train].copy()
        x_test = x_train_full.iloc[idx_test].copy()
        y_test = y_train[idx_test].copy()
        y_train = y_train[idx_train].copy()

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
    joblib.dump(pipeline, out_dir / "pipeline.joblib")

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
    project_root = Path(__file__).resolve().parent.parent

    default_xlsx = project_root / "dashboard" / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    default_out = project_root / "app" / "model"

    p = argparse.ArgumentParser(description="Treino longitudinal PEDE (t -> t+1).")

    p.add_argument(
        "--xlsx",
        type=str,
        default=str(default_xlsx),
        help="Caminho do XLSX.",
    )

    p.add_argument(
        "--out",
        type=str,
        default=str(default_out),
        help="Diretório de saída dos artefatos.",
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