"""Data loading helpers for churn modelling."""

from pathlib import Path


def load_customer_data(dataset_path: Path | str | None = None):
    """Load customer data from CSV or return a small demo dataset.

    The demo dataset keeps the project runnable for recruiters even when a private
    production dataset cannot be committed to the repository.
    """
    import pandas as pd

    if dataset_path is None:
        return build_demo_customer_data()

    path = Path(dataset_path)
    if not path.exists():
        msg = f"Dataset not found: {path}"
        raise FileNotFoundError(msg)

    return pd.read_csv(path)


def build_demo_customer_data():
    """Create a compact, realistic sample dataset for smoke tests and demos."""
    import pandas as pd

    records = [
        {
            "tenure_months": 2,
            "monthly_charges": 74.9,
            "support_tickets": 4,
            "contract_type": "month_to_month",
            "payment_method": "electronic_check",
            "churn": "yes",
        },
        {
            "tenure_months": 38,
            "monthly_charges": 64.2,
            "support_tickets": 0,
            "contract_type": "two_year",
            "payment_method": "credit_card",
            "churn": "no",
        },
        {
            "tenure_months": 12,
            "monthly_charges": 82.1,
            "support_tickets": 3,
            "contract_type": "month_to_month",
            "payment_method": "bank_transfer",
            "churn": "yes",
        },
        {
            "tenure_months": 55,
            "monthly_charges": 45.4,
            "support_tickets": 1,
            "contract_type": "one_year",
            "payment_method": "credit_card",
            "churn": "no",
        },
        {
            "tenure_months": 6,
            "monthly_charges": 89.7,
            "support_tickets": 5,
            "contract_type": "month_to_month",
            "payment_method": "electronic_check",
            "churn": "yes",
        },
        {
            "tenure_months": 48,
            "monthly_charges": 58.9,
            "support_tickets": 0,
            "contract_type": "two_year",
            "payment_method": "bank_transfer",
            "churn": "no",
        },
        {
            "tenure_months": 18,
            "monthly_charges": 70.3,
            "support_tickets": 2,
            "contract_type": "one_year",
            "payment_method": "credit_card",
            "churn": "no",
        },
        {
            "tenure_months": 4,
            "monthly_charges": 95.0,
            "support_tickets": 6,
            "contract_type": "month_to_month",
            "payment_method": "electronic_check",
            "churn": "yes",
        },
    ]
    return pd.DataFrame.from_records(records)
