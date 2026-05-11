"""Command-line interface for training the churn model."""

import argparse
from pathlib import Path

from churn_prediction.config import TrainingConfig
from churn_prediction.model import train_and_evaluate


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the training workflow."""
    parser = argparse.ArgumentParser(description="Train a customer churn prediction model.")
    parser.add_argument("--data", type=Path, default=None, help="Path to a CSV dataset.")
    parser.add_argument(
        "--target",
        default="churn",
        help="Name of the binary target column. Default: churn.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/churn_model.joblib"),
        help="Where the trained model will be saved.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size.")
    return parser.parse_args()


def main() -> None:
    """Train the model and print a concise evaluation summary."""
    args = parse_args()
    config = TrainingConfig(
        dataset_path=args.data,
        model_output_path=args.model_output,
        target_column=args.target,
        test_size=args.test_size,
    )

    metrics = train_and_evaluate(config)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
