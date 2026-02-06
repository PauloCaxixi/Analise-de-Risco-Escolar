import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

from src.preprocessing import load_raw_data, basic_cleaning

# Dados de referência (treino)
df_ref = load_raw_data("data/raw/BASE_DE_DADOS_PEDE_2024_DATATHON.xlsx")
df_ref = basic_cleaning(df_ref)

# Simulação de dados novos (produção)
df_prod = df_ref.sample(frac=0.3, random_state=42)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_ref, current_data=df_prod)

report.save_html("monitoring/drift_report.html")
