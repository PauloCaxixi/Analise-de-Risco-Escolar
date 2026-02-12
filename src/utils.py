import random
import numpy as np
import pandas as pd
from typing import Tuple


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def split_temporal(
    df: pd.DataFrame,
    year_column: str,
    test_year: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = df[df[year_column] != test_year]
    test_df = df[df[year_column] == test_year]
    return train_df, test_df
