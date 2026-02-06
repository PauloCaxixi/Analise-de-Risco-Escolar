import pandas as pd
from src.config import SHEETS, DROP_COLS


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Lê todas as sheets do Excel e consolida em um único DataFrame,
    mantendo rastreabilidade via coluna ANO_PEDE.
    """
    dfs = []

    for year, sheet_name in SHEETS.items():
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df["ANO_PEDE"] = year
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpeza básica:
    - Remove colunas irrelevantes
    - Padroniza nomes das colunas
    - Trata valores nulos simples
    """
    df = df.copy()

    # Padroniza nomes das colunas
    df.columns = (
        df.columns
        .str.upper()
        .str.strip()
        .str.replace(" ", "_")
    )

    # Remove colunas irrelevantes
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Tratamento simples de nulos
    for col in df.select_dtypes(include=["float", "int"]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("DESCONHECIDO")

    return df

def make_columns_unique(df):
    """
    Garante que todas as colunas do DataFrame tenham nomes únicos
    """
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values] = [
            f"{dup}_{i}" if i != 0 else dup
            for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

def clean_data(df):
    """
    Limpeza e padronização dos tipos de dados
    """

    # Remove linhas completamente vazias
    df = df.dropna(how="all")

    # Remove espaços extras nos nomes das colunas
    df.columns = df.columns.str.strip()

    # Padroniza colunas categóricas para string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    return df

