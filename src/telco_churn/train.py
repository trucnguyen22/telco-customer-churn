"""Model training and tracking: fit model and log a versioned run to MLflow.

Run from the repository root:

    python -m telco_churn.train [--data PATH]

Each execution produces one MLflow run (parameters, metrics, fitted
pipeline, "telco-churn" model version). The MLflow "production" alias is 
chosen to serve as 'Model endpoint'.
"""

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from telco_churn.data import load_raw
from telco_churn.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
    extract_target,
)

DEFAULT_DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "telco-churn"
REGISTERED_MODEL_NAME = "telco-churn"

TEST_SIZE = 0.2
RANDOM_STATE = 42
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def build_pipeline() -> Pipeline:
    """Build the feature-processing and classifier pipeline.

    This saved MLflow pipeline can be reused directly for scoring. The
    ``handle_unknown="ignore"`` setting keeps inference robust when a
    category appears that was not present in training.
    """
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(**RF_PARAMS))])


def evaluate(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Held-out metrics for a fitted pipeline."""
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def train(data_path: str | Path) -> dict[str, float]:
    """Run one training cycle, log it to MLflow, return test metrics."""
    raw = load_raw(data_path)

    # Training policy: exclude brand-new customers (tenure == 0). They have
    # not had the opportunity to churn, so their labels carry no signal.
    # Scoring applies no such filter; build_features handles them there.
    raw = raw[raw["tenure"] > 0]

    X = build_features(raw)
    y = extract_target(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_params(RF_PARAMS)
        mlflow.log_params({"test_size": TEST_SIZE, "n_training_rows": len(X_train)})
        mlflow.log_metrics(metrics)
        sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1]),
            input_example=X_train.head(5),
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the churn model and log the run to MLflow."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the raw customer CSV (default: %(default)s).",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    metrics = train(args.data)
    for name, value in metrics.items():
        print(f"{name:9s} {value:.3f}")


if __name__ == "__main__":
    main()
