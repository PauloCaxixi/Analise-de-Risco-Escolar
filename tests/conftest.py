from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

import dashboard.app as dashboard_app


def _make_minimal_xlsx(path: Path) -> None:
    # Dataset mínimo com colunas usadas por dashboard/app e intervenções
    df_2022 = pd.DataFrame(
        [
            {
                "RA": "111",
                "Nome": "Aluno A",
                "Turma": "T1",
                "Instituição de ensino": "Escola X",
                "INDE 22": 5.0,
                "Matem": 5.0,
                "Portug": 6.0,
                "Inglês": 6.0,
                "Pedra 20": 2,
                "Pedra 21": 2,
                "Pedra 22": 2,
                "IEG": 0.30,
                "IPS": 0.35,
                "Indicado": "Sim",
                "Rec Psicologia": "",
                "Rec Av1": "",
                "Rec Av2": "",
                "Rec Av3": "",
                "Rec Av4": "",
                "Ativo/ Inativo": "Inativo",
                "IAA": 0.40,
                "IDA": 0.30,
                "IPV": 0.20,
                "IAN": 0.10,
            },
            {
                "RA": "222",
                "Nome": "Aluno B",
                "Turma": "T1",
                "Instituição de ensino": "Escola X",
                "INDE 22": 7.5,
                "Matem": 8.0,
                "Portug": 7.0,
                "Inglês": 7.5,
                "Pedra 20": 4,
                "Pedra 21": 4,
                "Pedra 22": 4,
                "IEG": 0.70,
                "IPS": 0.70,
                "Indicado": "Não",
                "Rec Psicologia": "",
                "Rec Av1": "OK",
                "Rec Av2": "",
                "Rec Av3": "",
                "Rec Av4": "",
                "Ativo/ Inativo": "Ativo",
                "IAA": 0.80,
                "IDA": 0.70,
                "IPV": 0.60,
                "IAN": 0.50,
            },
        ]
    )

    # Mantém 2023/2024 com as mesmas colunas (só para compatibilidade de leitura)
    df_2023 = df_2022.copy()
    df_2024 = df_2022.copy()

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df_2022.to_excel(w, sheet_name="PEDE2022", index=False)
        df_2023.to_excel(w, sheet_name="PEDE2023", index=False)
        df_2024.to_excel(w, sheet_name="PEDE2024", index=False)


@pytest.fixture(scope="session", autouse=True)
def _env_and_paths(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("pede_tests")
    xlsx_path = tmp_dir / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    _make_minimal_xlsx(xlsx_path)

    os.environ["PEDE_XLSX_PATH"] = str(xlsx_path)
    os.environ["PEDE_DEFAULT_SHEET"] = "PEDE2022"

    # Patch paths do módulo (garante que use o XLSX gerado)
    dashboard_app.DATA_XLSX_PATH = xlsx_path

    # Patch store do plano de reforço para não escrever no repo
    dashboard_app.PLANO_REFORCO_STORE = tmp_dir / "intervencoes_plano_reforco.csv"


@pytest.fixture()
def client() -> Generator:
    dashboard_app.app.config.update(TESTING=True)
    with dashboard_app.app.test_client() as c:
        yield c