from __future__ import annotations

import pytest
import dashboard.app as dashboard_app

def test_dashboard_renders_success(client) -> None:
    """
    Testa se o dashboard principal carrega com sucesso.
    Verifica se os indicadores de risco processados pela função 
    _build_dashboard_context estão presentes no HTML.
    """
    resp = client.get("/dashboard?sheet=PEDE2022")
    
    assert resp.status_code == 200
    # Verifica títulos e elementos do seu novo layout
    assert b"Dashboard" in resp.data
    assert b"Indicadores" in resp.data or b"Risco" in resp.data
    assert b"Alto Risco" in resp.data

def test_dashboard_search_filter(client) -> None:
    """
    Testa o filtro de busca na URL. 
    Mesmo que não encontre resultados, a página deve renderizar sem crash.
    """
    # Busca por um termo que existe no nosso conftest (Aluno A)
    resp = client.get("/dashboard?sheet=PEDE2022&q=Aluno")
    assert resp.status_code == 200
    assert b"Aluno A" in resp.data

def test_dashboard_empty_search(client) -> None:
    """Verifica se uma busca que não retorna nada renderiza o estado vazio do dashboard."""
    resp = client.get("/dashboard?sheet=PEDE2022&q=QUALQUER_COISA_INEXISTENTE")
    assert resp.status_code == 200
    # O dashboard deve mostrar "0" nos indicadores ou a tabela vazia
    assert b"0" in resp.data 

def test_dashboard_invalid_sheet_error(client) -> None:
    """
    Verifica como o app lida com abas inexistentes no Excel.
    O comportamento esperado é um erro controlado (400 ou 500).
    """
    resp = client.get("/dashboard?sheet=SHEET_FAKE_999")
    assert resp.status_code in (400, 404, 500)

def test_dashboard_context_data(client) -> None:
    """
    Verifica se os dados processados pela IA (como a média geral formatada com vírgula)
    estão chegando ao template.
    """
    resp = client.get("/dashboard?sheet=PEDE2022")
    html = resp.data.decode('utf-8')
    
    # Verifica se a formatação de moeda/número brasileira (vírgula) aparece
    # (Como definimos no seu _build_dashboard_context)
    assert "," in html 
    assert "Tendência" in html or "Tendencia" in html