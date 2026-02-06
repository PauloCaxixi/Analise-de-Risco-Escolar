## 📌 Projeto Datathon – Associação Passos Mágicos

**Previsão de Risco de Defasagem Escolar**

---

## 1. Visão Geral do Projeto

### 🎯 Objetivo

Este projeto tem como objetivo desenvolver um **modelo preditivo capaz de estimar o risco de defasagem escolar** de alunos atendidos pela Associação Passos Mágicos, utilizando dados educacionais, pedagógicos e psicossociais provenientes da pesquisa PEDE (Pesquisa Extensiva do Desenvolvimento Educacional).

O modelo visa apoiar a tomada de decisão da instituição, permitindo **intervenções educacionais mais rápidas e direcionadas**, gerando impacto social positivo.

---

### 💡 Solução Proposta

Foi construída uma **pipeline completa de Machine Learning**, cobrindo todo o ciclo de vida do modelo:

* Pré-processamento e engenharia de dados
* Treinamento e validação
* Deploy via API
* Empacotamento com Docker
* Testes unitários
* Monitoramento de drift

---

## 2. Stack Tecnológica

* **Linguagem:** Python 3.10
* **Manipulação de Dados:** pandas, numpy
* **Machine Learning:** scikit-learn
* **API:** FastAPI
* **Serialização:** joblib
* **Testes:** pytest, pytest-cov
* **Empacotamento:** Docker
* **Monitoramento:** Evidently
* **Deploy:** Local (Docker)

---

## 3. Estrutura do Projeto

```bash
Tech_4
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── model.joblib
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│
├── api/
│   ├── main.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_model.py
│
├── monitoring/
│   └── drift_report.html
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 4. Instruções de Deploy

### 🔹 Pré-requisitos

* Docker instalado
* Python 3.10+ (opcional, para execução local)

---

### 🔹 Build da imagem Docker

```bash
docker build -t passos-magicos-api .
```

### 🔹 Execução do container

```bash
docker run -p 8000:8000 passos-magicos-api
```

A API ficará disponível em:

```
http://localhost:8000
```

Documentação automática:

```
http://localhost:8000/docs
```

---

## 5. Treinamento do Modelo

Para treinar o modelo localmente:

```bash
python src/train.py
```

O modelo treinado será salvo em:

```
models/model.joblib
```

---

## 6. Exemplo de Chamada à API

### 🔹 Endpoint

```
POST /predict
```

### 🔹 Exemplo de requisição

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "RA": 12345,
  "ANO_PEDE": 2024,
  "IDADE": 14,
  "FASE": "Fase 3",
  "TURMA": "3A",
  "INDE_2022": 6.2,
  "INDE_2023": 5.4,
  "INDE_2024": 5.1
}'
```

### 🔹 Exemplo de resposta

```json
{
  "risco_defasagem": true,
  "probabilidade": 0.81
}
```

---

## 7. Pipeline de Machine Learning

### 🔹 Etapas

1. **Pré-processamento:** limpeza, padronização e tratamento de nulos
2. **Engenharia de Features:** uso de indicadores educacionais históricos
3. **Criação da Target:** variável derivada de critérios educacionais (INDE)
4. **Treinamento:** Random Forest com balanceamento de classes
5. **Avaliação:** Precision, Recall e F1-score
6. **Deploy:** API FastAPI + Docker
7. **Monitoramento:** detecção de data drift com Evidently

---

## 8. Testes Unitários

Execução dos testes:

```bash
pytest --cov=src tests/
```

Cobertura mínima garantida: **≥ 80%**

---

## 9. Monitoramento de Drift

O monitoramento é realizado com **Evidently**, gerando um relatório HTML:

```
monitoring/drift_report.html
```

Esse painel permite identificar mudanças no comportamento dos dados ao longo do tempo.

---

## 📌 Conclusão

Este projeto entrega uma solução completa, escalável e alinhada às boas práticas de MLOps, com potencial real de impacto social na educação de crianças e jovens em situação de vulnerabilidade.

