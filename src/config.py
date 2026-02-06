from pathlib import Path

# Diretórios base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"

# Nome das sheets
SHEETS = {
    2022: "PEDE2022",
    2023: "PEDE2023",
    2024: "PEDE2024"
}

# Coluna identificadora do aluno
ID_COL = "RA"

# Colunas que NÃO devem entrar no modelo
DROP_COLS = [
    "NOME",  # anonimizado, mas irrelevante para predição
]

# Limiar para definir risco de defasagem (baseado no INDE)
INDE_RISK_THRESHOLD = 5.5
