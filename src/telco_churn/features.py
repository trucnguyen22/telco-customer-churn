"""Feature engineering: transform a raw customer extract into model inputs.

This module owns the single definition of the model's features, shared by
training and scoring so the two can never drift apart. The key contract of
:func:`build_features` is that it is row-preserving: every input row yields
exactly one output row. Deciding which rows are suitable for *training*
(e.g. excluding brand-new customers) is a policy that belongs to the
training job, not to feature computation.
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
    """Transform the raw extract into the model-ready feature frame.

    Row-preserving: every input row yields exactly one output row, so the
    result stays index-aligned with ``raw_df`` (and with
    :func:`extract_target`). Works with or without the ``Churn`` column,
    so the same code path serves training and scoring.

    Args:
        raw_df: Raw customer data as returned by ``data.load_raw``.
            Not modified.

    Returns:
        A frame containing exactly ``FEATURE_COLUMNS``, indexed like
        ``raw_df``.
    """
    df = raw_df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["num_addon_services"] = _count_addon_services(df)
    df["charges_delta"] = _compute_charges_delta(df)
    return df[FEATURE_COLUMNS]


def extract_target(raw_df: pd.DataFrame) -> pd.Series:
    """Return the binary churn target (1 = churned), index-aligned with
    the output of :func:`build_features`.
    """
    return (raw_df[TARGET_COLUMN] == "Yes").astype(int).rename("churn")


def _count_addon_services(df: pd.DataFrame) -> pd.Series:
    """Number of add-on services (0-6) each customer subscribes to."""
    return (df[ADDON_COLUMNS] == "Yes").sum(axis=1)


def _compute_charges_delta(df: pd.DataFrame) -> pd.Series:
    """Current monthly rate minus the customer's lifetime average rate.

    Rule: a customer with ``tenure == 0`` has no billing history, so the
    delta is defined as 0.0 (no deviation from a history that does not
    exist). This also makes the computation safe where the division
    would otherwise be undefined. Requires ``TotalCharges`` to already
    be numeric.
    """
    lifetime_avg = df["TotalCharges"] / df["tenure"]
    delta = (df["MonthlyCharges"] - lifetime_avg).round(2)
    return delta.where(df["tenure"] > 0, 0.0)
