"""Feature engineering utilities for the churn model."""

YES_NO_MAPPING = {
    "yes": 1,
    "y": 1,
    "true": 1,
    "1": 1,
    "no": 0,
    "n": 0,
    "false": 0,
    "0": 0,
}


def normalize_binary_target(value: object) -> int:
    """Convert common churn labels to 0 or 1 with a helpful error message."""
    normalized_value = str(value).strip().lower()
    if normalized_value not in YES_NO_MAPPING:
        msg = f"Unsupported target value: {value!r}. Use yes/no, true/false or 1/0."
        raise ValueError(msg)
    return YES_NO_MAPPING[normalized_value]


def split_features_and_target(dataframe, target_column: str):
    """Return feature matrix and normalized binary target vector."""
    if target_column not in dataframe.columns:
        msg = f"Target column '{target_column}' was not found in the dataset."
        raise KeyError(msg)

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column].map(normalize_binary_target)
    return features, target


def detect_column_types(dataframe) -> tuple[list[str], list[str]]:
    """Identify numeric and categorical columns for preprocessing."""
    numeric_columns = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in dataframe.columns if column not in numeric_columns]
    return numeric_columns, categorical_columns
