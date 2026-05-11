# customer-churn-prediction
Machine learning project to predict customer churn using Python, Pandas and Scikit-learn.
# Customer Churn Prediction

Projeto de machine learning para prever **churn de clientes** com uma pipeline limpa, testável e pronta para portfólio. A solução organiza carregamento de dados, preparação de features, treinamento, avaliação e persistência do modelo em módulos pequenos e fáceis de revisar.

## Por que este projeto é relevante para recrutadores?

Este repositório demonstra práticas esperadas em projetos profissionais de dados e backend Python:

- **Código modular**: responsabilidades separadas entre configuração, dados, features, modelo e CLI.
- **Pipeline reproduzível**: preprocessing e classificador ficam no mesmo artefato salvo, reduzindo risco de inconsistência entre treino e inferência.
- **Boas práticas de qualidade**: type hints, nomes claros, testes automatizados e configuração de lint.
- **Experiência de uso simples**: o projeto roda com um dataset próprio em CSV ou com um dataset demonstrativo embutido.
- **Foco em negócio**: métricas como ROC AUC e relatório de classificação ajudam a avaliar a capacidade do modelo de priorizar clientes com maior risco de churn.

## Estrutura do projeto

```text
.
├── pyproject.toml
├── README.md
├── src/
│   └── churn_prediction/
│       ├── cli.py          # Interface de linha de comando
│       ├── config.py       # Configuração validada do treinamento
│       ├── data.py         # Leitura de CSV e dataset demo
│       ├── features.py     # Normalização do alvo e detecção de colunas
│       └── model.py        # Pipeline, treino, avaliação e persistência
└── tests/
    └── test_features.py
```

## Tecnologias

- Python 3.10+
- Pandas
- Scikit-learn
- Joblib
- Pytest
- Ruff

## Como executar localmente

### 1. Criar e ativar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -e ".[dev]"
```

### 3. Rodar os testes

```bash
pytest
```

### 4. Treinar com o dataset demonstrativo

```bash
train-churn-model
```

O comando salva o modelo em `models/churn_model.joblib` e imprime métricas de avaliação no terminal.

### 5. Treinar com um CSV próprio

```bash
train-churn-model \
  --data data/customers.csv \
  --target churn \
  --model-output models/churn_model.joblib
```

O CSV deve conter uma coluna alvo binária. Por padrão, o projeto espera a coluna `churn`, aceitando valores como `yes/no`, `true/false` ou `1/0`.

## Como a solução funciona

1. **Carregamento dos dados**: lê um CSV informado pelo usuário ou usa um dataset demo pequeno para validação rápida.
2. **Preparação do alvo**: normaliza a variável de churn para `0` ou `1` com mensagens de erro explícitas para valores inválidos.
3. **Preprocessamento**:
   - colunas numéricas recebem imputação por mediana e padronização;
   - colunas categóricas recebem imputação por moda e one-hot encoding.
4. **Treinamento**: usa regressão logística com `class_weight="balanced"`, uma escolha interpretável e adequada como baseline profissional.
5. **Avaliação**: reporta acurácia, ROC AUC e relatório de classificação.
6. **Persistência**: salva a pipeline completa para reutilização consistente.

## Próximos passos recomendados

- Adicionar um dataset real versionado via DVC ou armazenado fora do Git.
- Criar um notebook de análise exploratória com insights de negócio.
- Comparar modelos adicionais, como Random Forest, Gradient Boosting e XGBoost.
- Adicionar validação cruzada e busca de hiperparâmetros.
- Expor o modelo em uma API FastAPI com endpoint de predição.
- Monitorar drift de dados e performance após implantação.

## Licença

Este projeto está licenciado sob os termos da licença MIT.
