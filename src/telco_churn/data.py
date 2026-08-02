"""Data loading: loading and schema validation for the raw customer extract.
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
    """Load the raw customer extract from a CSV file to pd.DataFrame.

    Args:
        path: Location of the raw CSV file.

    Returns:
        The raw data with all 21 expected columns, unmodified.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file's columns do not match ``RAW_COLUMNS``.
    """
    df = pd.read_csv(path, dtype_backend='numpy_nullable')
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` naming every missing or unexpected column."""
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    unexpected = sorted(set(df.columns) - set(RAW_COLUMNS))
    if missing or unexpected:
        raise ValueError(
            "Raw data does not match the expected schema. "
            f"Missing columns: {missing}. Unexpected columns: {unexpected}."
        )
