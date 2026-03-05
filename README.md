# 🎓 Dashboard Educacional — Passos Mágicos (Datathon)

Este projeto é uma solução de **Inteligência de Dados e Monitoramento Pedagógico** desenvolvida para o Datathon.
O sistema processa dados longitudinais de alunos, aplica modelos de **Machine Learning para prever riscos de defasagem** e oferece uma **interface de gestão pedagógica** para tomada de decisão rápida.

---

---

# 📈 Objetivo do Projeto

Este projeto busca aplicar **Ciência de Dados na Educação**, permitindo:

* identificar alunos em risco
* antecipar defasagens escolares
* apoiar decisões pedagógicas
* orientar intervenções educacionais

Tudo através de **dados longitudinais e inteligência artificial**.

---

# 🛠️ Arquitetura do Sistema (app.py)

O núcleo da aplicação foi construído em **Flask** e está dividido em camadas lógicas para garantir **robustez, legibilidade e escalabilidade**.

---

# 1️⃣ Bootstrap e Configuração

O código utiliza **Pathlib** para garantir que o projeto rode em qualquer sistema operacional (**Windows / Linux / MacOS**) sem problemas de caminho.

O sistema:

* Define o `REPO_ROOT`
* Injeta as pastas do projeto no `sys.path`
* Permite importar módulos internos de `src/`

Isso evita problemas comuns de importação quando o projeto roda fora da raiz.

---

# 2️⃣ Pipeline de Dados (Normalização)

Antes de qualquer análise ou predição, os dados passam por um pipeline de limpeza.

### `_standardize_columns`

Responsável por padronizar nomes de colunas:

* remove acentos
* remove espaços
* converte para **snake_case**
* garante compatibilidade com o modelo

Exemplo:

```
"Nota Matemática" → nota_matematica
```

---

### `_coerce_numeric`

Transforma colunas críticas em valores numéricos:

* Notas
* INDE
* Pedras

Se houver textos como:

```
"Mantido na Fase"
```

o sistema converte para:

```
NaN
```

Isso evita quebra nos cálculos matemáticos.

---

### `_read_xlsx_sheet`

Camada de **IO controlado** responsável por:

* Ler o arquivo `.xlsx`
* Validar se o arquivo existe
* Validar se a aba existe

Abas suportadas:

* 2022
* 2023
* 2024

Caso a aba não exista, o sistema gera erro controlado.

---

# 3️⃣ Motor de Predição de Risco

O sistema opera em modo **Híbrido**.

## 🧠 Modo ML (Oficial)

Carrega os arquivos:

```
model.joblib
preprocessor.joblib
```

Utiliza:

```
predict_proba()
```

para calcular a **probabilidade de defasagem educacional futura** com base nas features definidas em:

```
metadata.json
```

---

## 🛟 Modo Fallback (Heurístico)

Caso o modelo ainda não tenha sido treinado, o sistema ativa automaticamente:

```
_predict_risk_fallback()
```

Essa função calcula o risco com base em:

* INDE
* Médias atuais
* Histórico de evolução

Isso garante que **o dashboard nunca fique vazio**.

---

# 4️⃣ Inteligência Artificial Pedagógica

O sistema possui um motor de análise pedagógica automatizado.

### `gerar_recomendacao_ia`

Analisa:

* Pedras
* Risco
* Notas
* Perfil psicológico

E gera um **parecer textual automático** para o coordenador pedagógico.

Exemplo de saída:

```
Aluno apresenta risco moderado de defasagem.
Recomendado reforço em matemática e acompanhamento socioemocional.
```

---

### Detecção de Estagnação

A função:

```
detectar_alunos_sem_progresso()
```

cruza os dados de:

* 2022
* 2023
* 2024

Para detectar alunos que:

* não evoluíram de pedra
* mantiveram notas estagnadas por 2 anos

Esses casos são destacados no dashboard para intervenção.

---

# 🚀 Como Executar

## 1️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

# 🧠 Tecnologias Utilizadas

* Python
* Flask
* Pandas
* Scikit-learn
* Joblib
* Machine Learning
* Data Engineering

---

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



