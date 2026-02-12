import numpy as np
import pandas as pd
import json
from typing import Dict


def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = 10,
) -> float:
    expected = expected.dropna()
    actual = actual.dropna()

    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.percentile(expected, breakpoints)
    actual_perc = np.percentile(actual, breakpoints)

    psi = 0.0
    for i in range(buckets):
        e_pct = (
            ((expected >= expected_perc[i]) & (expected < expected_perc[i + 1])).mean()
        )
        a_pct = (
            ((actual >= actual_perc[i]) & (actual < actual_perc[i + 1])).mean()
        )

        if e_pct > 0 and a_pct > 0:
            psi += (a_pct - e_pct) * np.log(a_pct / e_pct)

    return psi


def compute_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> Dict[str, float]:
    drift_report = {}

    for feature in features:
        drift_report[feature] = calculate_psi(
            baseline[feature],
            current[feature],
        )

    return drift_report
