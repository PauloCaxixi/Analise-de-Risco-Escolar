import pandas as pd
from typing import Tuple, List
from src.config import TARGET_COLUMN

LEAKAGE_COLUMNS = [
    TARGET_COLUMN,
    "TARGET",
]

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop, errors="ignore")


def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    X = select_features(df)
    y = df["TARGET"]
    return X, y
