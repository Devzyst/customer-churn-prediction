"""Model training and evaluation workflow."""

from pathlib import Path
from typing import Any

from churn_prediction.config import TrainingConfig
from churn_prediction.data import load_customer_data
from churn_prediction.features import detect_column_types, split_features_and_target


def build_preprocessing_pipeline(features):
    """Create preprocessing steps for numeric and categorical customer attributes."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_columns, categorical_columns = detect_column_types(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_model_pipeline(features):
    """Build the complete estimator used for churn prediction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline(features)),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=42),
            ),
        ]
    )


def train_and_evaluate(config: TrainingConfig) -> dict[str, Any]:
    """Train the churn model, persist it and return core evaluation metrics."""
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
    from sklearn.model_selection import train_test_split

    config.validate()
    customer_data = load_customer_data(config.dataset_path)
    features, target = split_features_and_target(customer_data, config.target_column)

    stratify_target = target if target.nunique() > 1 else None
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_target,
    )

    model_pipeline = build_model_pipeline(train_features)
    model_pipeline.fit(train_features, train_target)

    predictions = model_pipeline.predict(test_features)
    churn_probabilities = model_pipeline.predict_proba(test_features)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(test_target, predictions), 4),
        "roc_auc": round(roc_auc_score(test_target, churn_probabilities), 4),
        "classification_report": classification_report(test_target, predictions, zero_division=0),
    }

    save_model(model_pipeline, config.model_output_path)
    return metrics


def save_model(model_pipeline, output_path: Path) -> None:
    """Persist a trained model pipeline to disk."""
    import joblib

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, output_path)
