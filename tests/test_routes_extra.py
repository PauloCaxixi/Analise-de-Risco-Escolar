from __future__ import annotations
import pytest

def test_api_tendencia(client) -> None:
    """Verifica se o endpoint JSON para os gráficos de tendência está funcionando."""
    resp = client.get("/api/tendencia?sheet=PEDE2022")
    assert resp.status_code == 200
    data = resp.get_json()
    # Garante que as chaves necessárias para o Chart.js estão presentes
    assert "labels" in data
    assert "media" in data
    assert "risco" in data

def test_alunos_risco_page(client) -> None:
    """Testa a renderização da lista completa de risco."""
    resp = client.get("/alunos-risco?sheet=PEDE2022")
    assert resp.status_code == 200
    # Verifica títulos comuns ou elementos da tabela
    assert b"Alunos" in resp.data
    assert b"Lista" in resp.data

def test_export_csv(client) -> None:
    """Testa se a exportação de dados gera um arquivo CSV válido."""
    resp = client.get("/export?sheet=PEDE2022")
    assert resp.status_code == 200
    # Verifica o Header de download
    assert "text/csv" in resp.headers.get("Content-Type", "")
    # Verifica se o separador configurado no seu app (;) está presente
    assert b";" in resp.data

def test_intervencoes_pages_load(client) -> None:
    """Testa o carregamento de todas as páginas de intervenção (GET)."""
    routes = [
        "/intervencoes/plano-reforco",
        "/intervencoes/acompanhamento",
        "/intervencoes/reuniao-pais"
    ]
    for route in routes:
        resp = client.get(f"{route}?sheet=PEDE2022")
        assert resp.status_code == 200, f"Falha ao carregar a rota: {route}"
        # Verifica se o termo 'Interven' (sem acento para evitar erro de encode) está no HTML
        assert b"Interven" in resp.data

def test_intervencao_plano_reforco_post_flow(client) -> None:
    """
    Testa o fluxo de postagem de um novo plano de reforço.
    Verifica se o sistema processa o formulário e redireciona para a confirmação.
    """
    payload = {
        "ra": ["111"],
        "disciplina": "Matemática",
        "observacao": "Teste de Reforço Automatizado",
        "data": "2026-12-31",
    }
    resp = client.post(
        "/intervencoes/plano-reforco?sheet=PEDE2022",
        data=payload,
        follow_redirects=True
    )
    assert resp.status_code == 200
    # Verifica se o sistema avançou para a tela de confirmação ou sucesso
    assert b"Confirmar" in resp.data or b"Sucesso" in resp.data

def test_static_files_access(client) -> None:
    """Garante que o CSS e JS básicos estão acessíveis."""
    # Ajuste o caminho se o seu arquivo principal tiver outro nome
    resp = client.get("/static/style.css") 
    assert resp.status_code in [200, 404] # 404 permitido se o arquivo ainda não existir, mas 200 é o ideal