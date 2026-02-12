from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from app.db import get_conn
import os

DATA_PATH = Path(os.getenv("PEDE_XLSX_PATH", "data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx")).resolve()


SHEETS = {
    "PEDE2022": {
        "name_cols": ["Nome"],
        "defasagem_col": "Defasagem",
    },
    "PEDE2023": {
        "name_cols": ["Nome Anonimizado"],
        "defasagem_col": "Defasagem",
    },
    "PEDE2024": {
        "name_cols": ["Nome Anonimizado"],
        "defasagem_col": "Defasagem",
    },
}


def _resolve_name(row: pd.Series, name_cols: list[str]) -> str:
    for c in name_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return "SEM_NOME"


def _normalize_features(row: pd.Series, def_col: str) -> Dict:
    feats = {}
    for col, val in row.items():
        if col == def_col:
            continue
        if pd.isna(val):
            continue
        # normaliza tipos simples
        if isinstance(val, (int, float, str, bool)):
            feats[col] = val
        else:
            feats[col] = str(val)
    return feats


def import_all() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Excel não encontrado em: {DATA_PATH}")

    with get_conn() as conn:

        cur = conn.cursor()

        for sheet, cfg in SHEETS.items():
            df = pd.read_excel(DATA_PATH, sheet_name=sheet)
            df.columns = [c.strip() for c in df.columns]

            if "RA" not in df.columns:
                raise ValueError(f"Sheet {sheet} sem coluna RA.")

            def_col = cfg["defasagem_col"]

            for _, row in df.iterrows():
                ra = str(row["RA"]).strip()
                if not ra:
                    continue

                name = _resolve_name(row, cfg["name_cols"])

                # upsert student por RA
                cur.execute("SELECT id FROM students WHERE student_code = ?", (ra,))
                r = cur.fetchone()
                if r:
                    student_id = int(r["id"])
                else:
                    cur.execute(
                        """
                        INSERT INTO students (student_code, full_name)
                        VALUES (?, ?)
                        """,
                        (ra, name),
                    )
                    student_id = int(cur.lastrowid)

                # features por ano (snapshot)
                features = _normalize_features(row, def_col)
                cur.execute(
                    """
                    INSERT INTO grades (student_id, reference_year, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (student_id, sheet, json.dumps(features, ensure_ascii=False)),
                )

                # defasagem real (ground truth)
                if def_col in row and pd.notna(row[def_col]):
                    cur.execute(
                        """
                        INSERT INTO predictions
                        (student_id, reference_year, risk_score, risk_label, model_version, input_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            student_id,
                            sheet,
                            float(row[def_col]),
                            "REAL",
                            "ground_truth",
                            json.dumps({}, ensure_ascii=False),
                        ),
                    )

        conn.commit()


if __name__ == "__main__":
    import_all()
    print("Importação PEDE concluída com sucesso.")
