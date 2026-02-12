from src.preprocessing import preprocess_all


def test_preprocess_returns_dataframe():
    df = preprocess_all()
    assert not df.empty
    assert "TARGET" in df.columns
