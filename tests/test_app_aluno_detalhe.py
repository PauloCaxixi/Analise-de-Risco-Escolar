from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path
import dashboard.app as dashboard_app

def _get_ra_from_test_file() -> str:
    """
    Busca um RA válido no arquivo temporário criado pelo conftest.
    Isso garante que o teste não dependa de arquivos externos.
    """
    path = dashboard_app.DATA_XLSX_PATH
    df = pd.read_excel(path, sheet_name="PEDE2022")
    return str(df["RA"].iloc[0])

def test_aluno_detalhe_success(client) -> None:
    """
    Testa se a página de detalhes carrega corretamente para um aluno existente.
    Verifica a presença dos novos elementos de IA e Recomendações.
    """
    ra = _get_ra_from_test_file()
    
    # Simula o acesso à rota
    resp = client.get(f"/aluno/{ra}?sheet=PEDE2022")
    
    assert resp.status_code == 200
    # Verifica elementos do novo template ajustado
    assert b"Ficha do Aluno" in resp.data or b"Detalhes do Aluno" in resp.data
    assert b"Diagnostico Pedagogico" in resp.data or b"Recomendacao da IA" in resp.data
    assert ra.encode() in resp.data

def test_aluno_detalhe_not_found(client) -> None:
    """Verifica se retorna 404 para um aluno que não consta na base."""
    resp = client.get("/aluno/999999999?sheet=PEDE2022")
    assert resp.status_code == 404

def test_aluno_detalhe_layout_elements(client) -> None:
    """Verifica se os cards de indicadores e ações estão presentes no HTML."""
    ra = _get_ra_from_test_file()
    resp = client.get(f"/aluno/{ra}?sheet=PEDE2022")
    
    html = resp.data.decode('utf-8')
    
    # Verifica se os blocos de lógica que inserimos no template aparecem
    assert "Risco (IA)" in html
    assert "Plano de Intervencao" in html
    assert "INDE 22" in html
    assert "Matem" in html

def test_aluno_detalhe_no_sheet_param(client) -> None:
    """Verifica o comportamento quando o parâmetro ?sheet está ausente (deve usar default)."""
    ra = _get_ra_from_test_file()
    resp = client.get(f"/aluno/{ra}")
    
    # Se o sistema estiver robusto, deve redirecionar ou carregar com a sheet padrão (200)
    assert resp.status_code in [200, 302]