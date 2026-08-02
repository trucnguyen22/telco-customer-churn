"""Feature engineering: transform a raw customer extract into model inputs.

Shared by training and scoring so the two can never drift apart.
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
    """Transform the raw extract pd.DataFrame into the model-ready feature pd.DataFrame.

    Args:
        raw_df: as returned by ``data.load_raw``.

    Returns:
        A pd.DataFrame containing ``FEATURE_COLUMNS``, 
            index-aligned with ``raw_df``.
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
    """The binary churn target (1 = churned), index-aligned with
        the output of `build_features`.
    """
    return (raw_df[TARGET_COLUMN] == "Yes").astype(int).rename("churn")


def _count_addon_services(df: pd.DataFrame) -> pd.Series:
    """Number of add-on services (0-6) each customer subscribes to."""
    return (df[ADDON_COLUMNS] == "Yes").sum(axis=1)


def _compute_charges_delta(df: pd.DataFrame) -> pd.Series:
    """Current monthly rate minus the customer's lifetime average rate.

    Rule: a customer with ``tenure == 0`` is defined as 0.0 (no billing history). 
        Requires ``TotalCharges`` to already be numeric.
    """
    lifetime_avg = df["TotalCharges"] / df["tenure"]
    delta = (df["MonthlyCharges"] - lifetime_avg).round(2)
    return delta.where(df["tenure"] > 0, 0.0)
