# customer-churn-prediction
Machine learning project to predict customer churn using Python, Pandas and Scikit-learn.
# Customer Churn Prediction

Projeto profissional de Machine Learning para prever a probabilidade de cancelamento de clientes (*customer churn*) a partir de dados cadastrais, contratuais e de uso de serviços.

Este repositório foi organizado para portfólio técnico: estrutura simples, código modular, comandos reproduzíveis e documentação suficiente para que um recrutador ou avaliador técnico entenda rapidamente o problema, a solução e como executar o pipeline.

## Objetivo

Construir um pipeline reproduzível para:

- carregar e limpar dados de clientes;
- separar variáveis explicativas e variável alvo;
- aplicar pré-processamento adequado para variáveis numéricas e categóricas;
- treinar um classificador de churn com `RandomForestClassifier`;
- avaliar o modelo com métricas de classificação;
- salvar o modelo treinado para uso posterior em predições em lote.

## Tecnologias usadas

- Python 3.10+
- Pandas
- Scikit-learn
- Joblib
- Makefile para comandos de conveniência

## Dataset utilizado

O repositório inclui um dataset amostral em `data/customer_churn.csv`, com 80 registros sintéticos inspirados em problemas reais de churn em telecomunicações. As principais variáveis são:

- informações demográficas: `gender`, `senior_citizen`, `partner`, `dependents`;
- informações de relacionamento: `tenure_months`, `contract`, `payment_method`;
- serviços contratados: `phone_service`, `internet_service`, `online_security`, `tech_support`;
- cobranças: `monthly_charges`, `total_charges`;
- alvo supervisionado: `Churn`.

O dicionário de dados está disponível em `data/README.md`.

> O dataset é pequeno e serve para demonstrar organização, boas práticas e execução ponta a ponta. Em um caso real, recomenda-se substituir esse arquivo por uma base histórica maior e revisar validação, drift e regras de negócio.

## Pipeline de Machine Learning

1. **Carregamento**: leitura do CSV com validação de existência do arquivo.
2. **Limpeza**: remoção de duplicidades, padronização de nomes de colunas, limpeza de textos vazios e normalização do alvo.
3. **Separação**: divisão entre features e target, removendo identificadores como `customer_id` para evitar memorização.
4. **Pré-processamento**: imputação e escala para variáveis numéricas; imputação e One-Hot Encoding para variáveis categóricas.
5. **Treino**: modelo `RandomForestClassifier` com pesos balanceados para lidar com classes desbalanceadas.
6. **Avaliação**: cálculo de métricas de classificação.
7. **Persistência**: salvamento do modelo em `model/churn_model.joblib` e resumo de métricas em `model/metrics.json`.

## Resultados

As métricas são calculadas automaticamente pelo script de treino e salvas em `model/metrics.json`.

| Métrica | Por que importa em churn |
| --- | --- |
| Accuracy | Mostra o percentual total de classificações corretas. |
| Precision | Ajuda a reduzir falsos positivos em campanhas de retenção. |
| Recall | Mede quantos clientes com risco de churn foram identificados. |
| F1-score | Equilibra precision e recall. |
| ROC AUC | Avalia a capacidade geral do modelo de separar churn de não churn. |

Para atualizar os resultados, execute:

```bash
python -m src.train
```

ou, se preferir:

```bash
make train
```

## Como rodar o projeto

### 1. Criar e ativar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

ou:

```bash
make install
```

### 3. Treinar o modelo

```bash
python -m src.train
```

Esse comando salva:

- modelo treinado em `model/churn_model.joblib`;
- métricas e metadados em `model/metrics.json`.

### 4. Gerar predições

```bash
python -m src.predict --input data/customer_churn.csv --output data/predictions.csv
```

O arquivo de saída contém as colunas originais mais:

- `predicted_churn`: classe prevista, em que `1` indica churn;
- `churn_probability`: probabilidade estimada de churn.

## Estrutura de pastas

```text
customer-churn-prediction/
├── data/
│   ├── README.md
│   └── customer_churn.csv
├── model/
│   ├── README.md
│   ├── churn_model.joblib       # criado pelo treino
│   └── metrics.json             # criado pelo treino
├── notebook/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
└── requirements.txt
```

## Principais arquivos

- `src/config.py`: caminhos e constantes centrais do projeto;
- `src/preprocess.py`: funções de carregamento, limpeza, remoção de identificadores, separação de alvo e criação do pré-processador;
- `src/train.py`: pipeline completo de treino, avaliação e persistência do modelo;
- `src/predict.py`: carregamento do modelo treinado e geração de predições em lote;
- `model/README.md`: descrição dos artefatos gerados pelo treinamento;
- `data/customer_churn.csv`: dataset amostral usado para demonstração;
- `data/README.md`: dicionário de dados;
- `requirements.txt`: dependências necessárias para execução;
- `Makefile`: atalhos para instalação, treino, predição, validação e limpeza de artefatos.

## Checklist técnico

- [x] Estrutura clara para projeto de ML.
- [x] Código modular e com responsabilidades separadas.
- [x] Pipeline de treino e inferência reproduzível.
- [x] Métricas relevantes para classificação binária.
- [x] Modelo salvo na pasta `model/` após treino.
- [x] README orientado a portfólio e recrutadores.
- [x] Dataset de exemplo e dicionário de dados versionados.

## Próximos passos sugeridos

- adicionar testes automatizados com `pytest`;
- incluir validação cruzada e busca de hiperparâmetros;
- versionar experimentos com MLflow ou DVC;
- publicar um notebook de análise exploratória em `notebook/`;
- disponibilizar uma API de inferência com FastAPI;
- monitorar performance e drift em produção.
