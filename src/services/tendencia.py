from __future__ import annotations

from typing import List, Tuple

import pandas as pd


PEDRAS = ["Pedra 20", "Pedra 21", "Pedra 22"]

def calcular_tendencia(df: pd.DataFrame) -> Tuple[List[float], List[int]]:
    """
    - media: média por coluna Pedra
    - risco: quantidade de alunos com Pedra <= 2 (proxy de risco)
    """
    tendencia_media: List[float] = []
    tendencia_risco: List[int] = []

    for pedra in PEDRAS:
        if pedra not in df.columns:
            tendencia_media.append(0.0)
            tendencia_risco.append(0)
            continue

        s = pd.to_numeric(df[pedra], errors="coerce")
        media = float(s.mean(skipna=True)) if s.notna().any() else 0.0
        risco = int((s <= 2).sum())

        tendencia_media.append(round(media, 2))
        tendencia_risco.append(risco)

    return tendencia_media, tendencia_risco