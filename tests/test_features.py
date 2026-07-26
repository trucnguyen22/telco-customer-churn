"""Tests locking in the feature-engineering rules (telco_churn.features).

Each test pins one behavior from notebook Parts 3/5 or one production
rule the notebook never needed (tenure == 0, scoring without labels).
"""

import pandas as pd

from telco_churn.features import (
    FEATURE_COLUMNS,
    build_features,
    extract_target,
)


def test_charges_delta_is_rate_minus_lifetime_average(make_raw):
    # 380 total / 4 months = 95 average; current 100 -> delta 5.0
    features = build_features(make_raw(tenure=4, MonthlyCharges=100.0, TotalCharges="380"))

    assert features.loc[0, "charges_delta"] == 5.0


def test_tenure_zero_gets_neutral_charges_delta(make_raw):
    # Brand-new customer: blank TotalCharges, no billing history.
    features = build_features(make_raw(tenure=0, TotalCharges=" "))

    assert features.loc[0, "charges_delta"] == 0.0


def test_num_addon_services_counts_only_yes(make_raw):
    features = build_features(
        make_raw(
            OnlineSecurity="Yes",
            OnlineBackup="Yes",
            DeviceProtection="No",
            TechSupport="Yes",
            StreamingTV="No internet service",
            StreamingMovies="No",
        )
    )

    assert features.loc[0, "num_addon_services"] == 3


def test_senior_citizen_recoded_to_yes_no(make_raw):
    assert build_features(make_raw(SeniorCitizen=0)).loc[0, "SeniorCitizen"] == "No"
    assert build_features(make_raw(SeniorCitizen=1)).loc[0, "SeniorCitizen"] == "Yes"


def test_output_has_exactly_the_feature_columns(make_raw):
    features = build_features(make_raw())

    assert list(features.columns) == FEATURE_COLUMNS


def test_every_row_survives(make_raw):
    raw = pd.concat(
        [make_raw(), make_raw(tenure=0, TotalCharges=" "), make_raw(tenure=72)],
        ignore_index=True,
    )

    assert len(build_features(raw)) == 3


def test_input_frame_is_not_mutated(make_raw):
    raw = make_raw()
    before = raw.copy(deep=True)

    build_features(raw)

    pd.testing.assert_frame_equal(raw, before)


def test_works_without_churn_column(make_raw):
    # Scoring extracts have no label; feature building must not need one.
    raw = make_raw().drop(columns=["Churn"])

    features = build_features(raw)

    assert list(features.columns) == FEATURE_COLUMNS


def test_extract_target_maps_yes_to_one(make_raw):
    assert extract_target(make_raw(Churn="Yes")).iloc[0] == 1
    assert extract_target(make_raw(Churn="No")).iloc[0] == 0
    assert extract_target(make_raw()).name == "churn"
