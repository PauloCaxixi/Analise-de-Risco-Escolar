from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

# 1º: CONFIGURAR CAMINHOS (Deve vir antes de qualquer import do projeto)
# Como conftest.py está em tech_5/Analise-de-Risco-Escolar/tests
# O ROOT deve ser tech_5/Analise-de-Risco-Escolar
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 2º: AGORA SIM, IMPORTAR OS MÓDULOS DO PROJETO
try:
    import dashboard.app as dashboard_app
except ModuleNotFoundError:
    # Fallback caso o pytest seja rodado de uma pasta acima
    ROOT_ALT = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT_ALT))
    import dashboard.app as dashboard_app

def _make_minimal_xlsx(path: Path) -> None:
    # Dataset mínimo com colunas usadas por dashboard/app e intervenções
    df_data = [
        {
            "RA": "111", "Nome": "Aluno A", "Turma": "T1",
            "Instituição de ensino": "Escola X", "INDE 22": 5.0,
            "Matem": 5.0, "Portug": 6.0, "Inglês": 6.0,
            "Pedra 20": 2, "Pedra 21": 2, "Pedra 22": 2,
            "IEG": 0.30, "IPS": 0.35, "Indicado": "Sim",
            "Rec Psicologia": "", "Rec Av1": "", "Rec Av2": "",
            "Rec Av3": "", "Rec Av4": "", "Ativo/ Inativo": "Inativo",
            "IAA": 0.40, "IDA": 0.30, "IPV": 0.20, "IAN": 0.10,
        },
        {
            "RA": "222", "Nome": "Aluno B", "Turma": "T1",
            "Instituição de ensino": "Escola X", "INDE 22": 7.5,
            "Matem": 8.0, "Portug": 7.0, "Inglês": 7.5,
            "Pedra 20": 4, "Pedra 21": 4, "Pedra 22": 4,
            "IEG": 0.70, "IPS": 0.70, "Indicado": "Não",
            "Rec Psicologia": "", "Rec Av1": "OK", "Rec Av2": "",
            "Rec Av3": "", "Rec Av4": "", "Ativo/ Inativo": "Ativo",
            "IAA": 0.80, "IDA": 0.70, "IPV": 0.60, "IAN": 0.50,
        }
    ]
    df = pd.DataFrame(df_data)

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="PEDE2022", index=False)
        df.to_excel(w, sheet_name="PEDE2023", index=False)
        df.to_excel(w, sheet_name="PEDE2024", index=False)

@pytest.fixture(scope="session", autouse=True)
def _env_and_paths(tmp_path_factory: pytest.TempPathFactory) -> None:
    tmp_dir = tmp_path_factory.mktemp("pede_tests")
    xlsx_path = tmp_dir / "BASE_TESTE.xlsx"
    _make_minimal_xlsx(xlsx_path)

    # Configura variáveis de ambiente para os testes
    os.environ["PEDE_XLSX_PATH"] = str(xlsx_path)
    os.environ["PEDE_DEFAULT_SHEET"] = "PEDE2022"

    # Patch nos paths globais do app para usar o diretório temporário
    dashboard_app.DATA_XLSX_PATH = xlsx_path
    dashboard_app.PLANO_REFORCO_STORE = tmp_dir / "intervencoes_plano_reforco.csv"

@pytest.fixture()
def client() -> Generator:
    dashboard_app.app.config.update(TESTING=True)
    with dashboard_app.app.test_client() as c:
        yield c