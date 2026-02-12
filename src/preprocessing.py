import pandas as pd
from typing import Dict
from src.config import (
    DATA_RAW_PATH,
    EXCEL_FILENAME,
    SHEETS,
    TARGET_COLUMN,
    TARGET_MAPPING,
)

def load_raw_data() -> Dict[str, pd.DataFrame]:
    file_path = DATA_RAW_PATH / EXCEL_FILENAME
    data = {}

    for sheet in SHEETS:
        df = pd.read_excel(file_path, sheet_name=sheet)
        df["ANO_REFERENCIA"] = sheet
        data[sheet] = df

    return data


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Coluna target '{TARGET_COLUMN}' não encontrada")

    df = df.copy()
    df["TARGET"] = df[TARGET_COLUMN].apply(TARGET_MAPPING)
    return df


def preprocess_all() -> pd.DataFrame:
    datasets = load_raw_data()
    processed_frames = []

    for _, df in datasets.items():
        df = create_target(df)
        processed_frames.append(df)

    full_df = pd.concat(processed_frames, axis=0, ignore_index=True)
    return full_df
