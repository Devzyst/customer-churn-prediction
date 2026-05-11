import pytest

from churn_prediction.config import TrainingConfig
from churn_prediction.features import normalize_binary_target


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("yes", 1),
        ("No", 0),
        (" TRUE ", 1),
        (0, 0),
        ("1", 1),
    ],
)
def test_normalize_binary_target_accepts_common_labels(raw_value, expected):
    assert normalize_binary_target(raw_value) == expected


def test_normalize_binary_target_rejects_unclear_labels():
    with pytest.raises(ValueError, match="Unsupported target value"):
        normalize_binary_target("maybe")


def test_training_config_requires_valid_test_size():
    with pytest.raises(ValueError, match="test_size"):
        TrainingConfig(test_size=1.2).validate()
