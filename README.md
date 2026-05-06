# customer-churn-prediction
Machine learning project to predict customer churn using Python, Pandas and Scikit-learn.
# Customer Churn Prediction

Projeto profissional de Machine Learning para prever a probabilidade de cancelamento de clientes (*customer churn*) a partir de dados cadastrais, contratuais e de uso de serviços.

## Objetivo

Construir um pipeline reproduzível para:

- carregar e limpar dados de clientes;
- separar variáveis explicativas e variável alvo;
- treinar um classificador de churn;
- avaliar o modelo com métricas de classificação;
- salvar o modelo treinado para uso posterior em predições em lote.

## Tecnologias usadas

- Python 3.10+
- Pandas
- Scikit-learn
- Joblib
- RandomForestClassifier

## Dataset utilizado

O repositório inclui um dataset amostral em `data/customer_churn.csv`, com 80 registros sintéticos inspirados em problemas reais de churn em telecomunicações. As principais variáveis são:

- informações demográficas: `gender`, `senior_citizen`, `partner`, `dependents`;
- informações de relacionamento: `tenure_months`, `contract`, `payment_method`;
- serviços contratados: `phone_service`, `internet_service`, `online_security`, `tech_support`;
- cobranças: `monthly_charges`, `total_charges`;
- alvo supervisionado: `Churn`.

> O dataset é pequeno e serve para demonstrar a organização do projeto, o pipeline de ML e a execução ponta a ponta. Em um caso real, recomenda-se substituir esse arquivo por uma base histórica maior.

## Resultados

As métricas são calculadas automaticamente pelo script de treino e salvas em `model/metrics.json`.

Pipeline utilizado:

1. carregamento do CSV;
2. limpeza de duplicidades, textos vazios e normalização do alvo;
3. separação entre treino e teste com estratificação;
4. pré-processamento de variáveis numéricas e categóricas;
5. treino com `RandomForestClassifier`;
6. avaliação com métricas de classificação.

Métricas reportadas pelo projeto:

| Métrica | Descrição |
| --- | --- |
| Accuracy | Percentual total de classificações corretas |
| Precision | Proporção de clientes previstos como churn que realmente cancelaram |
| Recall | Proporção de clientes churn corretamente identificados |
| F1-score | Média harmônica entre precision e recall |

Para atualizar os resultados, execute:

```bash
python src/train.py
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

### 3. Treinar o modelo

```bash
python src/train.py
```

Esse comando salva:

- modelo treinado em `model/churn_model.joblib`;
- métricas em `model/metrics.json`.

### 4. Gerar predições

```bash
python src/predict.py --input data/customer_churn.csv --output data/predictions.csv
```

O arquivo de saída contém as colunas originais mais:

- `predicted_churn`: classe prevista, em que `1` indica churn;
- `churn_probability`: probabilidade estimada de churn.

## Estrutura de pastas

```text
customer-churn-prediction/
├── data/
│   └── customer_churn.csv
├── model/
│   ├── churn_model.joblib       # criado pelo treino
│   └── metrics.json             # criado pelo treino
├── notebook/
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Principais arquivos

- `src/preprocess.py`: funções de carregamento, limpeza, separação de alvo e criação do pré-processador;
- `src/train.py`: pipeline completo de treino, avaliação e persistência do modelo;
- `src/predict.py`: carregamento do modelo treinado e geração de predições em lote;
- `data/customer_churn.csv`: dataset amostral usado para demonstração;
- `requirements.txt`: dependências necessárias para execução.

## Próximos passos sugeridos

- adicionar testes automatizados com `pytest`;
- incluir validação cruzada e busca de hiperparâmetros;
- versionar experimentos com MLflow ou DVC;
- publicar um notebook de análise exploratória em `notebook/`;
- disponibilizar uma API de inferência com FastAPI.
