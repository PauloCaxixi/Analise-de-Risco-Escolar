from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftResult:
    feature: str
    psi: float
    status: str  # OK | ATENCAO | DRIFT
    ref_count: int
    cur_count: int


def _safe_series(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out


def psi(
    reference: pd.Series,
    current: pd.Series,
    *,
    bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Population Stability Index (PSI).

    - reference: distribuição de treino (baseline)
    - current: distribuição atual (produção)

    PSI < 0.10 => OK
    0.10-0.25 => ATENÇÃO
    >= 0.25   => DRIFT

    Retorna PSI em float.
    """
    ref = _safe_series(reference)
    cur = _safe_series(current)

    if ref.empty or cur.empty:
        return float("nan")

    # Bins por quantis do reference (mais estável)
    try:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(ref.to_numpy(), quantiles))
        if edges.size < 3:  # poucos valores únicos
            return 0.0
    except Exception:
        return float("nan")

    ref_counts, _ = np.histogram(ref.to_numpy(), bins=edges)
    cur_counts, _ = np.histogram(cur.to_numpy(), bins=edges)

    ref_dist = ref_counts / max(ref_counts.sum(), 1)
    cur_dist = cur_counts / max(cur_counts.sum(), 1)

    ref_dist = np.clip(ref_dist, eps, None)
    cur_dist = np.clip(cur_dist, eps, None)

    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def psi_status(value: float) -> str:
    if np.isnan(value):
        return "SEM_DADOS"
    if value < 0.10:
        return "OK"
    if value < 0.25:
        return "ATENCAO"
    return "DRIFT"


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    features: Iterable[str],
    bins: int = 10,
) -> List[DriftResult]:
    """
    Calcula PSI para um conjunto de features.
    """
    results: List[DriftResult] = []
    for f in features:
        if f not in reference_df.columns or f not in current_df.columns:
            continue
        ref_s = reference_df[f]
        cur_s = current_df[f]
        value = psi(ref_s, cur_s, bins=bins)
        results.append(
            DriftResult(
                feature=f,
                psi=value if not np.isnan(value) else float("nan"),
                status=psi_status(value),
                ref_count=int(pd.to_numeric(ref_s, errors="coerce").dropna().shape[0]),
                cur_count=int(pd.to_numeric(cur_s, errors="coerce").dropna().shape[0]),
            )
        )
    # Ordena por maior PSI (mais crítico primeiro)
    results.sort(key=lambda r: (np.isnan(r.psi), -r.psi if not np.isnan(r.psi) else -1.0))
    return results


def summarize(results: List[DriftResult]) -> Dict[str, int]:
    out = {"OK": 0, "ATENCAO": 0, "DRIFT": 0, "SEM_DADOS": 0}
    for r in results:
        out[r.status] = out.get(r.status, 0) + 1
    return out