"""Tests for the batch scoring logic (telco_churn.score)."""

import pandas as pd

from telco_churn.features import build_features, extract_target
from telco_churn.score import score_frame
from telco_churn.train import _build_pipeline


def _fitted_pipeline(make_raw):
    """A real pipeline fitted on two tiny synthetic customers."""
    train_raw = pd.concat(
        [make_raw(Churn="Yes"), make_raw(Churn="No", Contract="Two year")],
        ignore_index=True,
    )
    pipeline = _build_pipeline()
    pipeline.fit(build_features(train_raw), extract_target(train_raw))
    return pipeline


def test_scores_every_customer_including_brand_new(make_raw):
    pipeline = _fitted_pipeline(make_raw)
    raw = pd.concat(
        [
            make_raw(customerID="0001-OLD"),
            make_raw(customerID="0002-NEW", tenure=0, TotalCharges=" "),
        ],
        ignore_index=True,
    )

    scores = score_frame(pipeline, raw, threshold=0.5)

    assert list(scores["customerID"]) == ["0001-OLD", "0002-NEW"]
    assert scores["churn_probability"].between(0.0, 1.0).all()


def test_flag_agrees_with_threshold(make_raw):
    pipeline = _fitted_pipeline(make_raw)
    raw = pd.concat([make_raw(), make_raw(tenure=60)], ignore_index=True)

    scores = score_frame(pipeline, raw, threshold=0.5)

    expected = scores["churn_probability"] >= 0.5
    assert (scores["churn_flag"] == expected).all()
