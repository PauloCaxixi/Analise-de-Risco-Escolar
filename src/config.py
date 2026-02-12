from pathlib import Path

# =========================
# Paths do projeto
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "app" / "model"

# =========================
# Dataset
# =========================
EXCEL_FILENAME = "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
SHEETS = ["PEDE2022", "PEDE2023", "PEDE2024"]
TARGET_COLUMN = "Defasagem"

# =========================
# Modelagem
# =========================
RANDOM_STATE = 42
TEST_YEAR = "PEDE2024"

RISK_THRESHOLD_LOW = 0.4
RISK_THRESHOLD_HIGH = 0.7

# =========================
# Target engineering
# =========================
# Defasagem > 0 => risco
TARGET_MAPPING = lambda x: 1 if x > 0 else 0
