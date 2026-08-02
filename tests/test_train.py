"""Tests for the training pipeline definition (telco_churn.train)."""

import pandas as pd

from telco_churn.features import build_features, extract_target
from telco_churn.train import _build_pipeline


def test_pipeline_survives_unseen_categories(make_raw):
    # Fit on two customers, then score one whose InternetService value
    # never appeared in training. handle_unknown="ignore" must make this
    # a valid prediction, not a crash.
    train_raw = pd.concat(
        [make_raw(Churn="Yes"), make_raw(Churn="No", Contract="Two year")],
        ignore_index=True,
    )
    pipeline = _build_pipeline()
    pipeline.fit(build_features(train_raw), extract_target(train_raw))

    unseen = build_features(make_raw(InternetService="Quantum Fiber 9000"))
    proba = pipeline.predict_proba(unseen)[:, 1]

    assert 0.0 <= proba[0] <= 1.0
