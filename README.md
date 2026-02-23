```md
README.md:
# Dashboard Educacional — Passos Mágicos (Datathon)

Dashboard de monitoramento de risco de reprovação com:
- Menu lateral e topbar (busca + escola + notificações + usuário)
- Cards de indicadores (alto risco, risco médio, regulares, média geral com gauge)
- Tabela de alunos em alto risco (com "Ver Detalhes")
- Card "Filtrar por Disciplina"
- Gráfico "Tendência Mensal" (Pedra 20/21/22)
- Seção "Ações Recomendadas"
- Card "Próximos Prazos" (config manual)

> A classificação de risco é lida do modelo (quando existir) ou calculada por fallback (heurística) para o dashboard não quebrar.

---

## 1) Pré-requisitos

- Python 3.10+
- (Opcional) Docker

---

## 2) Estrutura

```

.
├── app.py
├── requirements.txt
├── Dockerfile
├── templates/
│   ├── base.html
│   ├── home.html
│   └── aluno_detalhe.html
└── static/
└── css/
└── style.css

````

---

## 3) Rodar local (Python)

### Instalar dependências
```bash
pip install -r requirements.txt
````

### Definir caminho do XLSX (obrigatório)

Você precisa apontar para o arquivo:

* `BASE DE DADOS PEDE 2024 - DATATHON.xlsx`

No Windows (PowerShell):

```powershell
$env:PEDE_XLSX_PATH="C:\caminho\para\BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
python app.py
```

No Linux/macOS:

```bash
export PEDE_XLSX_PATH="/caminho/para/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
python app.py
```

### Acessar

* Dashboard: `http://localhost:5000/dashboard`

### Trocar a sheet (2022/2023/2024)

Por querystring:

* `http://localhost:5000/dashboard?sheet=PEDE2022`
* `http://localhost:5000/dashboard?sheet=PEDE2023`
* `http://localhost:5000/dashboard?sheet=PEDE2024`

---

## 4) Filtros (escola e busca)

### Busca global (topo)

Campo busca:

* Nome
* Turma
* Matem / Portug / Inglês (conteúdo textual/número)

### Escola

O seletor exibido vem da coluna:

* `Instituição de ensino`

> No código, o filtro por escola está preparado via querystring `?escola=...`.

Exemplo:

* `http://localhost:5000/dashboard?sheet=PEDE2022&escola=Escola%20X`

---

## 5) Rodar com Docker

### Build

```bash
docker build -t pede-dashboard .
```

### Run

Passe o caminho do XLSX via bind mount + env var.

**Windows (PowerShell)**:

```powershell
docker run --rm -p 5000:5000 `
  -e PEDE_XLSX_PATH="C:\data\BASE DE DADOS PEDE 2024 - DATATHON.xlsx" `
  -e PEDE_DEFAULT_SHEET="PEDE2022" `
  pede-dashboard
```

**Linux/macOS**:

```bash
docker run --rm -p 5000:5000 \
  -e PEDE_XLSX_PATH="/data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" \
  -e PEDE_DEFAULT_SHEET="PEDE2022" \
  pede-dashboard
```

---

Troque este código:

```md
## 6) Modelo (opcional)

Se você treinar um modelo e salvar em:

```

app/model/model.joblib
app/model/preprocessor.joblib
app/model/metadata.json

````

o dashboard automaticamente passa a usar o modelo.

### metadata.json esperado (mínimo)
```json
{
  "feature_columns": ["INDE 22", "IEG", "IPS", "Matem", "Portug", "Inglês"]
}
````

Se não existir modelo/metadata, o sistema usa fallback (heurística) para manter a UI funcionando.

````

por este código:
```md
## 6) Treinamento do Modelo (obrigatório no Datathon)

O treino é **longitudinal (t -> t+1)**:
- Treino: `PEDE2022 -> PEDE2023`
- Teste temporal: `PEDE2023 -> PEDE2024` (quando houver RA em comum)

### Instalar dependências (inclui scikit-learn)
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
````

### Rodar treino (gera artefatos em `app/model/`)

```bash
python -m src.train --xlsx "CAMINHO/BASE DE DADOS PEDE 2024 - DATATHON.xlsx" --out "app/model"
```

Arquivos gerados:

```
app/model/model.joblib
app/model/preprocessor.joblib
app/model/metadata.json
```

### Métricas reportadas

O treino calcula (quando houver conjunto de teste temporal válido):

* `roc_auc`
* `pr_auc`
* `f1`
* `precision`
* `recall`

> Para o contexto educacional, **recall** é prioridade: perder aluno em risco é pior do que falso positivo.

### Como o Dashboard usa o modelo

Ao iniciar o `app.py`, se os arquivos abaixo existirem:

* `app/model/model.joblib`
* `app/model/preprocessor.joblib`
* `app/model/metadata.json` com `feature_columns`

o dashboard passa a usar:

* `predict_proba` → `P(defasagem)` (0 a 1)
* thresholds → `Muito Alto / Alto / Médio / Regular`

Thresholds (configurados em `metadata.json`):

* `>= 0.85` → Muito Alto
* `>= 0.70` → Alto
* `>= 0.50` → Médio
* `< 0.50` → Regular

Se os artefatos não existirem, o sistema usa fallback (heurística) apenas para manter a UI funcionando.

```

**Onde alterar:** no `README.md`, substitua a seção inteira **“## 6) Modelo (opcional)”** pelo bloco acima.

Confirma para eu enviar o próximo arquivo.

👉 Próximo sugerido: `src/__init__.py` (para permitir `python -m src.train`) + `src/features.py` (centralizar normalização/seleção de features e evitar duplicação entre `app.py` e `src/train.py`).
```


## 7) Rotas principais

### Dashboard (UI)
* `GET /dashboard` — tela principal
* `GET /aluno/<ra>` — tela de detalhes do aluno

---

### API de Predição (Datathon)

#### `POST /predict`

Endpoint oficial para inferência de risco de reprovação.

O endpoint aceita **1 aluno** ou **lista de alunos** e retorna:
- score de risco (`0.0 – 1.0`)
- classe de risco (`Muito Alto | Alto | Médio | Regular`)
- indicação se o modelo treinado foi usado ou fallback heurístico

---

#### 🔹 Exemplo 1 — Predição por RA (enriquecimento automático)

```bash
curl -X POST http://localhost:5000/predict?sheet=PEDE2022 \
  -H "Content-Type: application/json" \
  -d '{
    "RA": "123456"
  }'


Resposta:

{
  "model_version": "v1.0.0",
  "used_model": true,
  "predictions": [
    {
      "ra": "123456",
      "risk_score": 0.82,
      "risk_label": "Alto"
    }
  ]
}



## 8) Observações importantes

* O dashboard foi feito para ser **idêntico** ao layout especificado.
* O card “Próximos Prazos” é **manual** (não vem da planilha).
* A tendência mensal usa `Pedra 20`, `Pedra 21`, `Pedra 22` como eixo temporal.

---

```

Confirma para eu enviar o próximo arquivo.

👉 Próximo sugerido: `static/img/user.png` (fallback simples) **ou** `pytest` (tests + coverage >= 80%) para cumprir o requisito do datathon.
```
