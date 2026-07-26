"""Loading and schema validation for the raw Telco customer extract.

This module is the boundary between storage and the pipeline: every job
(training or scoring) obtains its input DataFrame through :func:`load_raw`,
which fails fast with a clear error if the file does not match the schema
the rest of the pipeline was built against.
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

    No cleaning is performed here: the returned frame is exactly what
    storage contains (e.g. ``TotalCharges`` may arrive as text because
    the export uses blank strings for missing values). Cleaning is the
    responsibility of ``features.build_features``.

    Args:
        path: Location of the raw CSV file.

    Returns:
        The raw data with all 21 expected columns, unmodified.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file's columns do not match ``RAW_COLUMNS``.
    """
    df = pd.read_csv(path)
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
