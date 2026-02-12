from src.preprocessing import preprocess_all
from src.feature_engineering import split_features_target


def test_split_features_target():
    df = preprocess_all()
    X, y = split_features_target(df)

    assert len(X) == len(y)
    assert "TARGET" not in X.columns
