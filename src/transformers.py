from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ToNumeric(TransformerMixin, BaseEstimator):
    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        out = {c: pd.to_numeric(df[c], errors="coerce") for c in df.columns}
        return pd.DataFrame(out).to_numpy(dtype=float)


class ToString(TransformerMixin, BaseEstimator):
    def fit(self, X: Any, y: Any = None):
        return self

    def transform(self, X: Any):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        def _to_clean_str(v: Any):
            if pd.isna(v):
                return v
            return str(v).strip()

        df = df.map(_to_clean_str)
        return df.to_numpy(dtype=object)
