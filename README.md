
```markdown
# 🎓 Dashboard Educacional — Passos Mágicos (Datathon)

Este projeto é uma solução de **Inteligência de Dados e Monitoramento Pedagógico** desenvolvida para o Datathon. O sistema processa dados longitudinais de alunos, aplica modelos de Machine Learning para prever riscos de defasagem e oferece uma interface de gestão para intervenções rápidas.

---

## 🛠️ Arquitetura do Sistema (app.py)

O núcleo da aplicação foi construído em **Flask** e está dividido em camadas lógicas para garantir robustez e escalabilidade:

### 1. Bootstrap e Configuração
O código utiliza `Pathlib` para garantir que o projeto rode em qualquer sistema operacional (Windows/Linux) sem erro de caminhos. Ele configura o `REPO_ROOT` e injeta as pastas no `sys.path`, permitindo que os módulos internos em `src/` sejam importados corretamente.

### 2. Pipeline de Dados (Normalização)
* **`_standardize_columns`**: Limpa os nomes das colunas vindas do Excel (remove espaços, acentos e padroniza para snake_case/camelCase esperado).
* **`_coerce_numeric`**: Transforma colunas críticas (Notas, INDE, Pedras) em números reais. Se houver texto como "Mantido na Fase", ele converte para `NaN` para não quebrar os cálculos matemáticos.
* **`_read_xlsx_sheet`**: Camada de IO que lê o arquivo `.xlsx` oficial, validando se o arquivo e a aba (2022, 2023 ou 2024) existem.

### 3. Motor de Predição de Risco
O sistema opera em modo **Híbrido**:
* **Modo ML (Oficial)**: Carrega o `model.joblib` e o `preprocessor.joblib`. Utiliza `predict_proba` para calcular a probabilidade de defasagem futura baseada nas features do `metadata.json`.
* **Modo Fallback (Heurístico)**: Se o modelo não estiver treinado, o sistema aciona a função `_predict_risk_fallback`, que calcula o risco baseado no **INDE (Índice de Desenvolvimento Educacional)** e médias atuais, garantindo que o dashboard nunca fique vazio.

### 4. Inteligência Artificial Pedagógica
* **`gerar_recomendacao_ia`**: Uma lógica de IA interna que analisa o perfil do aluno (Pedras, Risco, Notas e Psicologia) e gera um parecer descritivo em texto natural para o coordenador.
* **Detecção de Estagnação**: A função `detectar_alunos_sem_progresso` cruza os dados de 2022 a 2024 para identificar alunos que não evoluíram de "Pedra" ou nota por 2 anos consecutivos.

---

## 🚀 Como Executar

### 1. Instalação
```bash
pip install -r requirements.txt

```

### 2. Configuração do Banco de Dados

O sistema lê o caminho da planilha através de variáveis de ambiente:

```powershell
# Windows
$env:PEDE_XLSX_PATH="C:\dados\BASE DE DADOS PEDE 2024 - DATATHON.xlsx"
# Linux/Mac
export PEDE_XLSX_PATH="/dados/BASE DE DADOS PEDE 2024 - DATATHON.xlsx"

```

### 3. Treinamento do Modelo

Para que o Dashboard use Machine Learning em vez da heurística:

```bash
python -m src.train --xlsx "caminho_da_base.xlsx" --out "app/model"

```

---

## 📡 API Reference (Endpoints)

| Rota | Método | Descrição |
| --- | --- | --- |
| `/dashboard` | `GET` | Interface principal com indicadores e filtros. |
| `/aluno/<ra>` | `GET` | Ficha detalhada do aluno com diagnóstico de IA. |
| `/predict` | `POST` | **Endpoint Datathon**: Recebe JSON com dados do aluno e retorna score/classe de risco. |
| `/export` | `GET` | Gera um CSV (com BOM para Excel) dos alunos em risco. |
| `/api/tendencia` | `GET` | Retorna dados JSON para os gráficos de evolução histórica. |

---

## 📊 Lógica de Negócio e Dashboards

### Filtros Inteligentes

O dashboard permite filtrar por **Instituição de Ensino** e **Busca Global** (Nome, RA ou Turma). Se uma escola for selecionada e não houver dados na aba atual, o sistema faz um fallback automático para "Todas as Escolas", evitando telas de erro 404.

### Réguas de Risco (Thresholds)

As cores e alertas do sistema seguem a lógica:

* 🔴 **Muito Alto**: Score ≥ 0.85 ou INDE < 4.0
* 🟠 **Alto**: Score ≥ 0.70 ou INDE < 5.5
* 🟡 **Médio**: Score ≥ 0.50 ou INDE < 6.5
* 🟢 **Regular**: Score < 0.50 ou INDE ≥ 6.5

---

## 🐳 Docker

Para rodar em container, utilize o `Dockerfile` incluso:

```bash
docker build -t pm-dash .
docker run -p 5000:5000 -e PEDE_XLSX_PATH="/app/data.xlsx" -v /caminho/local.xlsx:/app/data.xlsx pm-dash

```

---

**Nota:** Este projeto foi desenvolvido focado em **Recall**, garantindo que nenhum aluno com alta probabilidade de defasagem passe despercebido pela equipe pedagógica.

```

---

