# Script PowerShell para rodar testes + cobertura no Windows
# Uso:
#   .\scripts\run_cov.ps1
#
# Pré-requisitos:
#   - Python 3.10+
#   - venv (opcional)
#   - requirements.txt e requirements-dev.txt instalados

Write-Host "==> Ativando ambiente (se existir)..." -ForegroundColor Cyan
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

Write-Host "==> Instalando dependências (dev)..." -ForegroundColor Cyan
pip install -r requirements.txt | Out-Null
pip install -r requirements-dev.txt | Out-Null

Write-Host "==> Executando testes com cobertura..." -ForegroundColor Cyan
pytest --cov=dashboard --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Testes falharam." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Testes concluídos com sucesso." -ForegroundColor Green
Write-Host "📊 Relatório HTML gerado em: htmlcov/index.html" -ForegroundColor Yellow