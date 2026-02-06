import pandas as pd
from src.preprocessing import load_raw_data, basic_cleaning


def test_load_raw_data():
    df = load_raw_data("data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx")
    assert isinstance(df, pd.DataFrame)
    assert "ANO_PEDE" in df.columns
    assert len(df) > 0


def test_basic_cleaning():
    df = load_raw_data("data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx")
    df_clean = basic_cleaning(df)

    assert df_clean.isnull().sum().sum() == 0
