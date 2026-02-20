from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

import dashboard.app as dashboard_app


@pytest.fixture(scope="session", autouse=True)
def _set_env() -> None:
    # Ajuste: garante que o app use o XLSX do projeto (se existir ao lado do app.py)
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


def test_dashboard_renders(client) -> None:
    resp = client.get("/dashboard?sheet=PEDE2022")
    assert resp.status_code == 200
    assert b"Dashboard - Alunos com Alto Risco de Reprova" in resp.data


def test_dashboard_search_filter(client) -> None:
    # Busca genérica que não deve quebrar (mesmo que não encontre nada)
    resp = client.get("/dashboard?sheet=PEDE2022&q=turma")
    assert resp.status_code == 200
    assert b"Lista de Alunos em Alto Risco" in resp.data


def test_dashboard_invalid_sheet_returns_500(client) -> None:
    # Sheet inválida deve gerar erro interno (padrão atual do app)
    resp = client.get("/dashboard?sheet=SHEET_INEXISTENTE")
    assert resp.status_code in (400, 500)