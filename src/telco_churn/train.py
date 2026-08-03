"""Model training and tracking: fit model and log a versioned run to MLflow.

Run from the repository root:

    python -m telco_churn.train [--data PATH] [--promote]

Each execution produces one MLflow run (parameters, metrics, fitted
pipeline, "telco-churn" model version). The MLflow "production" alias is 
chosen to serve as 'Model endpoint'.
"""

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient
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
from telco_churn.config import DATA_PATH, EXPERIMENT_NAME, MODEL_NAME, PRODUCTION_ALIAS, TRACKING_URI

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _build_pipeline() -> Pipeline:
    """Create the preprocessing and modeling pipeline.

    The pipeline standardizes numeric features, one-hot encodes categorical
    features, and configures a RandomForestClassifier for churn prediction.

    Returns:
        Pipeline: An unfitted scikit-learn pipeline ready to be trained.
    """
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return Pipeline([
        ("pre", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))])


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute classification metrics for a fitted pipeline on test data.

    Args:
        pipeline: A fitted scikit-learn pipeline with ``predict`` and
            ``predict_proba`` methods.
        X_test: Feature values for the held-out test set.
        y_test: True labels for the held-out test set.

    Returns:
        Dict[str, float]: Metrics with keys ``accuracy``, ``precision``,
        ``recall``, ``f1``, and ``roc_auc``.
    """
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def train(data_path: str | Path, random_state=42, test_size=0.2, promote: bool = False) -> dict[str, float]:
    """Train a churn model, evaluate it, and log the run to MLflow.

    The function loads the raw dataset, engineers features, splits the data
    into train and test sets, trains the pipeline, evaluates the model, and
    logs the results to MLflow.

    Args:
        data_path: Path to the raw customer CSV file.
        random_state: Random seed for the train/test split and model setup.
        test_size: Fraction of rows reserved for the test set.
        promote: If True, attempt to promote the model version produced by
            this run to the production alias in the MLflow registry.

    Returns:
        Dict[str, float]: Classification metrics returned by ``_evaluate``.
    """

    # Load data.
    raw = load_raw(data_path)
    raw = raw[raw["tenure"] > 0]
    logger.info("loaded %d rows from %s", len(raw), data_path)

    # Feature, Prepare data
    X = build_features(raw)
    y = extract_target(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Build, Train, Evaluate pipeline
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = _evaluate(pipeline, X_test, y_test)

    # MLflow Tracking
    with mlflow.start_run() as run:
        mlflow.log_metrics(metrics)
        sklearn.log_model(
            pipeline,
            name="Random Forest",
            signature=infer_signature(X_train, pipeline.predict_proba(X_train)[:, 1]),
            input_example=X_train.head(5),
            registered_model_name=MODEL_NAME,
        )

    if promote:
        client = MlflowClient()
        versions = client.search_model_versions(
            f"name='{MODEL_NAME}' and run_id='{run.info.run_id}'"
        )
        if versions:
            version = versions[0].version
            client.set_registered_model_alias(MODEL_NAME, PRODUCTION_ALIAS, version)
            logger.info("promoted version %s to alias '%s'", version, PRODUCTION_ALIAS)
        else:
            logger.warning(
                "could not find registered model version for run %s; skipping promotion",
                run.info.run_id,
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the churn model and log the run to MLflow."
    )
    parser.add_argument(
        "--data",
        default=DATA_PATH,
        help="Path to the raw customer CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote the newly trained model version to the production alias.",
    )
    args = parser.parse_args()

    # MLflow setup (uri, experiment, runs)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # MLflow runs and Training process
    metrics = train(args.data, promote=args.promote)
    
    for name, value in metrics.items():
        print(f"{name:9s} {value:.3f}")


if __name__ == "__main__":
    main()
