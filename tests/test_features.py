from src.preprocessing import load_raw_data, basic_cleaning
from src.features import create_target


def test_create_target():
    df = load_raw_data("data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx")
    df = basic_cleaning(df)
    df = create_target(df)

    assert "RISCO_DEFASAGEM" in df.columns
    assert df["RISCO_DEFASAGEM"].isin([0, 1]).all()
