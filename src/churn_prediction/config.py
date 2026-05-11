"""Central configuration for the churn prediction pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values used by the training workflow."""

    dataset_path: Path | None = None
    model_output_path: Path = Path("models/churn_model.joblib")
    target_column: str = "churn"
    test_size: float = 0.2
    random_state: int = 42

    def validate(self) -> None:
        """Raise clear errors when a configuration value is invalid."""
        if not 0 < self.test_size < 1:
            msg = "test_size must be between 0 and 1."
            raise ValueError(msg)
        if not self.target_column.strip():
            msg = "target_column cannot be empty."
            raise ValueError(msg)
