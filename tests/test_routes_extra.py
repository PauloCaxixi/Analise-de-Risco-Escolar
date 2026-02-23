from __future__ import annotations

import pytest


def test_api_tendencia(client) -> None:
    resp = client.get("/api/tendencia?sheet=PEDE2022")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "labels" in data
    assert "media" in data
    assert "risco" in data


def test_alunos_risco_page(client) -> None:
    resp = client.get("/alunos-risco?sheet=PEDE2022")
    assert resp.status_code == 200
    assert b"Alunos - Lista Completa" in resp.data


def test_export_csv(client) -> None:
    resp = client.get("/export?sheet=PEDE2022")
    assert resp.status_code == 200
    assert resp.headers.get("Content-Type", "").startswith("text/csv")
    assert b"RA;" in resp.data  # CSV sep=;


def test_intervencao_plano_reforco_get(client) -> None:
    resp = client.get("/intervencoes/plano-reforco?sheet=PEDE2022")
    assert resp.status_code == 200
    assert b"Interven\xc3\xa7\xc3\xa3o - Plano de Refor\xc3\xa7o" in resp.data


def test_intervencao_plano_reforco_post_confirm(client) -> None:
    # Confirma 1 aluno (111)
    resp = client.post(
        "/intervencoes/plano-reforco?sheet=PEDE2022",
        data={
            "ra": ["111"],
            "disciplina": "Matemática",
            "observacao": "reforco",
            "data": "2026-01-01",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Confirmar Plano de Refor\xc3\xa7o" in resp.data


def test_intervencao_acompanhamento_get(client) -> None:
    resp = client.get("/intervencoes/acompanhamento?sheet=PEDE2022")
    assert resp.status_code == 200
    assert b"Interven\xc3\xa7\xc3\xa3o - Acompanhamento" in resp.data


def test_intervencao_reuniao_pais_get(client) -> None:
    resp = client.get("/intervencoes/reuniao-pais?sheet=PEDE2022")
    assert resp.status_code == 200
    assert b"Interven\xc3\xa7\xc3\xa3o - Reuni\xc3\xb5es de Pais" in resp.data