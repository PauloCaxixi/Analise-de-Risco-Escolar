import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


TARGET = "RISCO_DEFASAGEM"


def split_data(df):
    """
    Separa X e y usando todas as colunas disponíveis,
    exceto a variável alvo
    """
    if TARGET not in df.columns:
        raise ValueError(f"Coluna alvo '{TARGET}' não encontrada no dataframe.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_pipeline(X):
    """
    Cria pipeline automaticamente baseado nos tipos das colunas
    """

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    # 🔢 Pipeline numérico
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # 🔤 Pipeline categórico
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=42,
                class_weight="balanced"
            )),
        ]
    )

    return pipeline


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
