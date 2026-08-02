"""Feature engineering for the churn model.

This module transforms raw customer data into the feature matrix consumed by
training and scoring pipelines. The transformations are centralized here so
both workflows use identical logic.
"""

import pandas as pd

ADDON_COLUMNS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES: list[str] = [
    "tenure",
    "MonthlyCharges",
    "num_addon_services",
    "charges_delta",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COLUMN: str = "Churn"


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create a model-ready feature matrix from a raw customer extract.

    The input dataframe is copied, derived features are added, and the returned
    dataframe contains exactly the columns required for model training or
    inference.

    Args:
        raw_df: Raw customer extract as returned by ``data.load_raw``.

    Returns:
        DataFrame containing the engineered feature columns defined in
        ``FEATURE_COLUMNS``, aligned to the input rows.

    Raises:
        ValueError: If any engineered feature contains null values.
    """
    df = raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["num_addon_services"] = _count_addon_services(df)
    df["charges_delta"] = _compute_charges_delta(df)

    features = df[FEATURE_COLUMNS]
    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Feature engineering produced nulls in: {bad}")
    return features


def extract_target(raw_df: pd.DataFrame) -> pd.Series:
    """Return the binary churn target aligned with engineered features.

    Args:
        raw_df: Raw customer extract containing the ``Churn`` column.

    Returns:
        Series of 0/1 values named ``churn``, aligned to the input rows.
    """
    return (raw_df[TARGET_COLUMN] == "Yes").astype(int).rename("churn")


def _count_addon_services(df: pd.DataFrame) -> pd.Series:
    """Count the number of add-on services enabled by each customer.

    Args:
        df: DataFrame containing the add-on service columns listed in
            ``ADDON_COLUMNS``.

    Returns:
        Series of counts ranging from 0 to 6.
    """
    return (df[ADDON_COLUMNS] == "Yes").sum(axis=1)


def _compute_charges_delta(df: pd.DataFrame) -> pd.Series:
    """Compute the difference between monthly charges and lifetime-average charges.

    Customers with zero tenure are treated as having no billing history and
    receive a delta of 0.0.

    Args:
        df: DataFrame containing ``TotalCharges``, ``MonthlyCharges``, and
            ``tenure``.

    Returns:
        Series of rounded charge deltas.
    """
    lifetime_avg = df["TotalCharges"] / df["tenure"]
    delta = (df["MonthlyCharges"] - lifetime_avg).round(2)
    return delta.where(df["tenure"] > 0, 0.0)
