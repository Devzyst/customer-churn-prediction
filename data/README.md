# Data dictionary

O arquivo `customer_churn.csv` é um dataset sintético e versionável para demonstrar o pipeline ponta a ponta.

| Coluna | Tipo | Descrição |
| --- | --- | --- |
| `customer_id` | categórica | Identificador do cliente; removido antes do treino para evitar memorização. |
| `gender` | categórica | Gênero informado pelo cliente. |
| `senior_citizen` | categórica | Indica se o cliente é idoso. |
| `partner` | categórica | Indica se o cliente possui parceiro(a). |
| `dependents` | categórica | Indica se o cliente possui dependentes. |
| `tenure_months` | numérica | Tempo de relacionamento em meses. |
| `phone_service` | categórica | Indica contratação de serviço telefônico. |
| `internet_service` | categórica | Tipo de serviço de internet contratado. |
| `online_security` | categórica | Indica contratação de segurança online. |
| `tech_support` | categórica | Indica contratação de suporte técnico. |
| `contract` | categórica | Tipo de contrato. |
| `paperless_billing` | categórica | Indica faturamento digital. |
| `payment_method` | categórica | Método de pagamento. |
| `monthly_charges` | numérica | Valor mensal cobrado. |
| `total_charges` | numérica | Valor total acumulado. |
| `Churn` | alvo | Variável alvo: `Yes` para churn e `No` para retenção. |
