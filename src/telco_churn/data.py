"""Utilities for loading and validating the raw customer dataset.

This module provides a simple interface for reading the CSV extract used by
training and scoring workflows and for ensuring that its schema matches the
expected churn-model inputs.
"""

from pathlib import Path

import pandas as pd

RAW_COLUMNS: list[str] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw customer extract from a CSV file.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame containing the raw customer data with the expected schema.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the CSV columns do not match ``RAW_COLUMNS``.
    """
    df = pd.read_csv(path, dtype_backend="numpy_nullable")
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that a loaded dataframe has the expected schema.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required columns are missing or unexpected.
    """
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(RAW_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "Raw data does not match the expected schema. "
            f"Missing columns: {missing}. Unexpected columns: {unexpected}."
        )
