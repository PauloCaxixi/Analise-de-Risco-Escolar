
# 🎓 Monitor de Risco Escolar — Dashboard + Machine Learning

Este projeto foi desenvolvido para o **Datathon Educacional** e tem como objetivo aplicar **Ciência de Dados e Machine Learning na identificação precoce de risco de defasagem escolar**.

O sistema analisa dados educacionais históricos (2022–2024), calcula indicadores pedagógicos e utiliza um **modelo de Machine Learning para prever risco de reprovação**, apresentando os resultados em um **dashboard interativo para gestores educacionais**.

Além do dashboard, o projeto também expõe uma **API REST para consulta e predição de risco de alunos**.

---

# 🎯 Objetivo do Projeto

Aplicar **Data Science na Educação** para apoiar decisões pedagógicas baseadas em dados.

O sistema permite:

* identificar alunos em risco de reprovação
* antecipar defasagens educacionais
* analisar evolução longitudinal de desempenho
* orientar intervenções pedagógicas
* apoiar coordenadores e gestores escolares

---

# 🧠 Arquitetura do Sistema

O sistema é dividido em quatro camadas principais:

### 1️⃣ Camada de Dados

Responsável por leitura, limpeza e preparação dos dados educacionais.

### 2️⃣ Camada de Machine Learning

Treinamento do modelo e geração de predições de risco.

### 3️⃣ Camada de Aplicação

Dashboard interativo construído em **Flask**.

### 4️⃣ Camada de API

Endpoints REST para consulta e predição de risco de alunos.

---

# 📁 Estrutura do Projeto

Estrutura real do repositório:

```
Analise-de-Risco-Escolar
│
├── app
│   ├── metadata.json
│   ├── model.joblib
│   ├── pipeline.joblib
│   └── preprocessor.joblib
│
├── dashboard
│   ├── data
│   │   ├── processed
│   │   │   └── intervencoes_plano_reforco.csv
│   │   │
│   │   └── raw
│   │       └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
│   │
│   ├── app.py
│   └── drift_dashboard.py
│
├── src
│   ├── services
│   │   └── tendencia.py
│   │
│   ├── drift.py
│   ├── features.py
│   └── train.py
│
├── static
│   └── css
│       └── style.css
│
├── templates
│   ├── aluno_detalhe.html
│   ├── aluno_risco.html
│   ├── api_docs.html
│   ├── base.html
│   ├── home.html
│   ├── intervencao_acompanhamento.html
│   └── intervencao_reuniao_pais.html
│
├── tests
│   ├── conftest.py
│   ├── test_api_predict.py
│   ├── test_app_aluno_detalhe.py
│   ├── test_app_dashboard.py
│   └── test_routes_extra.py
│
├── Dockerfile
├── Makefile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Utilizado

O sistema utiliza uma planilha contendo dados educacionais estruturados em abas anuais:

```
PEDE2022
PEDE2023
PEDE2024
```

Localização do dataset:

```
dashboard/data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx
```

Cada linha representa um aluno com informações como:

* RA
* Nome
* Turma
* INDE (Índice de Desenvolvimento Educacional)
* Pedras pedagógicas
* Notas (Matemática, Português, Inglês)
* Indicadores educacionais (IEG, IPS, IAA, IDA, IPV, IAN)

Esses dados permitem realizar **análises longitudinais de desempenho escolar**.

---

# ⚙️ Pipeline de Processamento de Dados

Antes da análise ou predição, os dados passam por um pipeline de preparação.

### Leitura de dados

Função responsável:

```
_read_xlsx_sheet()
```

Localização:

```
dashboard/app.py
```

Responsável por:

* carregar o dataset
* validar a existência do arquivo
* validar a aba solicitada

---

### Padronização de colunas

Função:

```
_standardize_columns()
```

Localização:

```
src/features.py
```

Responsável por:

* padronizar nomes de colunas
* remover inconsistências de formatação
* garantir compatibilidade com o modelo

---

### Conversão numérica

Função:

```
_coerce_numeric()
```

Converte colunas críticas para formato numérico, evitando erros em cálculos estatísticos e no modelo.

---

# 🤖 Modelo de Machine Learning

O modelo é treinado utilizando **Scikit-learn** e armazenado no diretório:

```
app/
```

Arquivos gerados:

```
model.joblib
preprocessor.joblib
pipeline.joblib
metadata.json
```

O pipeline contém:

* tratamento de dados
* imputação de valores faltantes
* transformação de features
* modelo de classificação

---

# 🧠 Predição de Risco

A função principal de predição é:

```
_predict_risk_with_model()
```

Fluxo de predição:

```
dados do aluno
        ↓
preprocessor.transform()
        ↓
model.predict_proba()
        ↓
score de risco
```

O score varia entre **0 e 1**.

Classificação utilizada:

| Score  | Classificação |
| ------ | ------------- |
| ≥ 0.85 | Muito Alto    |
| ≥ 0.70 | Alto          |
| ≥ 0.50 | Médio         |
| < 0.50 | Regular       |

---

# 🛟 Sistema de Fallback

Caso o modelo não esteja disponível, o sistema ativa automaticamente:

```
_predict_risk_fallback()
```

Este método estima risco com base em:

* INDE
* médias das disciplinas
* indicadores educacionais

Isso garante que o dashboard continue funcionando mesmo sem modelo treinado.

---

# 📊 Dashboard Educacional

O dashboard apresenta indicadores importantes para gestão pedagógica:

* número de alunos em alto risco
* risco médio
* alunos regulares
* média geral da escola
* disciplinas com maior dificuldade
* tendência histórica de desempenho

Principais páginas:

```
/dashboard
/aluno/<ra>
/alunos-risco
```

---

# 📡 API REST

O sistema expõe endpoints REST para integração com outros sistemas.

### Health Check

```
GET /api/health
```

Retorna status da aplicação.

---

### Buscar aluno

```
GET /api/aluno/<ra>
```

Retorna informações completas de um aluno.

---

### Predição de risco

```
POST /api/predict
```

Entrada:

```json
{
  "RA": "12345"
}
```

Saída:

```json
{
  "ra": "12345",
  "risk_score": 0.72,
  "risk_label": "Alto"
}
```

---

### Predição em lote

```
POST /api/predict-batch
```

Entrada:

```json
[
  {"RA": "12345"},
  {"RA": "67890"}
]
```

---

### Tendência histórica

```
GET /api/tendencia
```

Retorna dados para gráficos de evolução histórica.

---

# 🧪 Testes Automatizados

Os testes estão localizados em:

```
tests/
```

Principais testes:

```
test_api_predict.py
test_app_dashboard.py
test_app_aluno_detalhe.py
```

Para executar os testes:

```
pytest
```

---

# 🚀 Como Executar o Projeto

### 1️⃣ Instalar dependências

```
pip install -r requirements.txt
```

---

### 2️⃣ Garantir que o dataset esteja em

```
dashboard/data/raw/BASE DE DADOS PEDE 2024 - DATATHON.xlsx
```

---

### 3️⃣ Executar o dashboard

Dentro da pasta `dashboard`:

```
python app.py
```

A aplicação iniciará em:

```
http://localhost:5000
```

---

# 🧠 Treinamento do Modelo

Para treinar o modelo de Machine Learning:

```
python src/train.py
```

Os arquivos gerados serão salvos em:

```
app/
```

Arquivos gerados:

```
model.joblib
preprocessor.joblib
pipeline.joblib
metadata.json
```

---

# 🧰 Tecnologias Utilizadas

* Python
* Flask
* Pandas
* Scikit-learn
* Joblib
* HTML
* CSS
* JavaScript
* Pytest

---

# 🎯 Resultado

A solução permite:

* monitorar risco de reprovação escolar
* identificar padrões educacionais
* apoiar decisões pedagógicas
* aplicar Machine Learning em dados educacionais reais

>>>>>>> Projeto_Monitoramento_Prevenção
