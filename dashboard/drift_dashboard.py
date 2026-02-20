from __future__ import annotations

import os
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.drift import DriftResult, compute_drift, summarize
from src.features import coerce_numeric, standardize_columns


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Monitoramento de Drift - Passos Mágicos",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_XLSX = BASE_DIR / "data" / "raw" / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
XLSX_PATH = Path(os.environ.get("PEDE_XLSX_PATH", str(DEFAULT_XLSX)))

DEFAULT_REF_SHEET = os.environ.get("PEDE_DRIFT_REF", "PEDE2022")
DEFAULT_CUR_SHEET = os.environ.get("PEDE_DRIFT_CUR", "PEDE2024")

FEATURES_MONITORED = [
    "INDE 22",
    "IEG",
    "IPS",
    "IDA",
    "IPV",
    "IAN",
    "Pedra 20",
    "Pedra 21",
    "Pedra 22",
    "Matem",
    "Portug",
    "Inglês",
]


# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_sheet(sheet: str) -> pd.DataFrame:
    if not XLSX_PATH.exists():
        st.error(f"Arquivo XLSX não encontrado: {XLSX_PATH}")
        st.stop()

    try:
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet, engine=None)
    except Exception as exc:
        st.error(f"Erro ao ler a sheet '{sheet}': {exc}")
        st.stop()

    df = standardize_columns(df)
    df = coerce_numeric(df)
    return df


def status_color(status: str) -> str:
    return {
        "OK": "🟢",
        "ATENCAO": "🟠",
        "DRIFT": "🔴",
        "SEM_DADOS": "⚪",
    }.get(status, "⚪")


# =========================
# UI
# =========================
st.title("📉 Monitoramento de Drift do Modelo")
st.caption("Population Stability Index (PSI) — acompanhamento contínuo das distribuições")

with st.sidebar:
    st.header("Configuração")
    ref_sheet = st.selectbox("Base de Referência (treino)", ["PEDE2022", "PEDE2023", "PEDE2024"], index=0)
    cur_sheet = st.selectbox("Base Atual (produção)", ["PEDE2022", "PEDE2023", "PEDE2024"], index=2)

    bins = st.slider("Número de bins (PSI)", min_value=5, max_value=20, value=10)

    st.markdown("---")
    st.markdown("**Interpretação do PSI**")
    st.markdown("- < 0.10 → OK")
    st.markdown("- 0.10–0.25 → ATENÇÃO")
    st.markdown("- ≥ 0.25 → DRIFT")

# =========================
# LOAD DATA
# =========================
df_ref = load_sheet(ref_sheet)
df_cur = load_sheet(cur_sheet)

# =========================
# COMPUTE DRIFT
# =========================
results: List[DriftResult] = compute_drift(
    df_ref,
    df_cur,
    features=FEATURES_MONITORED,
    bins=bins,
)

summary = summarize(results)

# =========================
# SUMMARY CARDS
# =========================
c1, c2, c3, c4 = st.columns(4)
c1.metric("🟢 OK", summary.get("OK", 0))
c2.metric("🟠 Atenção", summary.get("ATENCAO", 0))
c3.metric("🔴 Drift", summary.get("DRIFT", 0))
c4.metric("⚪ Sem Dados", summary.get("SEM_DADOS", 0))

st.markdown("---")

# =========================
# TABLE
# =========================
st.subheader("Detalhe por Feature")

table_rows = []
for r in results:
    table_rows.append(
        {
            "Status": f"{status_color(r.status)} {r.status}",
            "Feature": r.feature,
            "PSI": round(r.psi, 4) if not pd.isna(r.psi) else None,
            "Registros (Ref)": r.ref_count,
            "Registros (Atual)": r.cur_count,
        }
    )

df_table = pd.DataFrame(table_rows)
st.dataframe(
    df_table,
    use_container_width=True,
    hide_index=True,
)

# =========================
# DISTRIBUTION PLOTS
# =========================
st.markdown("---")
st.subheader("Distribuição (Referência × Atual)")

selected_feature = st.selectbox("Selecione a feature", FEATURES_MONITORED)

if selected_feature in df_ref.columns and selected_feature in df_cur.columns:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Referência**")
        fig_ref, ax_ref = plt.subplots()
        ax_ref.hist(
            df_ref[selected_feature].dropna(),
            bins=bins,
            alpha=0.8
        )
        ax_ref.set_title("Distribuição - Referência")
        ax_ref.set_xlabel(selected_feature)
        ax_ref.set_ylabel("Frequência")
        st.pyplot(fig_ref)

    with col2:
        st.markdown("**Atual**")
        fig_cur, ax_cur = plt.subplots()
        ax_cur.hist(
            df_cur[selected_feature].dropna(),
            bins=bins,
            alpha=0.8
        )
        ax_cur.set_title("Distribuição - Atual")
        ax_cur.set_xlabel(selected_feature)
        ax_cur.set_ylabel("Frequência")
        st.pyplot(fig_cur)
else:
    st.info("Feature indisponível nas bases selecionadas.")

st.caption(
    "Este painel atende ao requisito de monitoramento contínuo do Datathon, "
    "permitindo identificar quando o modelo deve ser reavaliado ou retreinado."
)