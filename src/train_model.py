import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import sys

# Resolve caminho do diretório atual (src/)
CURRENT_DIR = Path(__file__).resolve().parent

# Caminho do projeto raiz (Analise-de-Risco-Escolar)
PROJECT_DIR = CURRENT_DIR.parent

# Corrige o sys.path para apontar sempre para o projeto certo
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]  # .../Analise-de-Risco-Escolar
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.features import standardize_columns, calc_media_disciplinas

PROJECT_DIR = Path(__file__).resolve().parents[1]   # Analise-de-Risco-Escolar
DATA_PATH = PROJECT_DIR / "dashboard" / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

def load_sheet(sheet_name: str):
    df = pd.read_excel(DATA_PATH, sheet_name=sheet_name)

    # Normaliza colunas
    df = standardize_columns(df)

    # Remove colunas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    # Reseta índice e remove duplicados de índice
    df = df.reset_index(drop=True)

    return df


df2022 = load_sheet("PEDE2022")
df2023 = load_sheet("PEDE2023")
df2024 = load_sheet("PEDE2024")

# Concatenar SEM ERRO
df = pd.concat([df2022, df2023, df2024], ignore_index=True)

# 2) Criar média das disciplinas
df["_media"] = df.apply(calc_media_disciplinas, axis=1)

# 3) Seleção das features utilizadas pelo modelo
features = [
    "Matem", "Portug", "Inglês",
    "Pedra 20", "Pedra 21", "Pedra 22",
    "INDE 22", "IPS", "IEG",
    "_media", "ano"
]

df = df[features + ["INDE 22", "IPS", "Pedra 22"]].copy()

# 4) Criar target real de risco futuro (alta defasagem)
df["risco_futuro"] = (
    (df["INDE 22"] < 5) |
    (df["Pedra 22"] <= 2) |
    (df["IPS"] < 0.3)
).astype(int)

X = df[features]
y = df["risco_futuro"]

# 5) Treino do modelo
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)

# 6) Salvar modelo
joblib.dump(model, "app/model/model.joblib")

# 7) Salvar metadata
metadata = {
    "feature_columns": features,
    "model_version": "2.0-multianual",
    "target": "risco_futuro"
}

with open("app/model/metadata.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(metadata, ensure_ascii=False))