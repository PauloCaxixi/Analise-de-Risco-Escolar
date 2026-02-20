from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

import pandas as pd
import pytest

import dashboard.app as dashboard_app


def _xlsx_path() -> Optional[Path]:
    p = os.environ.get("PEDE_XLSX_PATH")
    if p:
        path = Path(p)
        return path if path.exists() else None
    # fallback: arquivo na raiz do repo
    base_dir = Path(__file__).resolve().parents[1]
    path = base_dir / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    return path if path.exists() else None


def _get_any_ra(sheet: str) -> Optional[str]:
    xlsx = _xlsx_path()
    if xlsx is None:
        return None
    df = pd.read_excel(xlsx, sheet_name=sheet, engine=None)
    if "RA" not in df.columns:
        return None
    s = df["RA"].dropna().astype(str).str.strip()
    s = s[s != ""]
    if s.empty:
        return None
    return str(s.iloc[0])


@pytest.fixture(scope="session", autouse=True)
def _set_env() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    xlsx_default = base_dir / "BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
    if xlsx_default.exists():
        os.environ["PEDE_XLSX_PATH"] = str(xlsx_default)
    os.environ.setdefault("PEDE_DEFAULT_SHEET", "PEDE2022")


@pytest.fixture()
def client() -> Generator:
    dashboard_app.app.config.update(TESTING=True)
    with dashboard_app.app.test_client() as c:
        yield c


def test_aluno_detalhe_renders_with_real_ra(client) -> None:
    ra = _get_any_ra("PEDE2022")
    if ra is None:
        pytest.skip("Sem XLSX disponível ou sem RA na sheet PEDE2022.")

    resp = client.get(f"/aluno/{ra}?sheet=PEDE2022")
    assert resp.status_code == 200
    assert b"Detalhes do Aluno" in resp.data
    assert ra.encode("utf-8") in resp.data


def test_aluno_detalhe_not_found(client) -> None:
    resp = client.get("/aluno/RA_INEXISTENTE_999?sheet=PEDE2022")
    assert resp.status_code == 404