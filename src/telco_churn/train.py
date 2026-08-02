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

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

def _build_pipeline() -> Pipeline:
    """Build a preprocessing and modeling pipeline.
    
    Creates a scikit-learn Pipeline that combines:
    - Numeric features: StandardScaler normalization
    - Categorical features: OneHotEncoder with drop='first'
    - Fitted RandomForestClassifier with predefined parameters
    
    Returns:
        Pipeline: A fitted preprocessing and classification pipeline.
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
    """Evaluate pipeline performance on test data.
    
    Computes classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    for the given pipeline on test features and target.
    
    Args:
        pipeline: A fitted scikit-learn Pipeline for classification.
        X_test: Test feature matrix.
        y_test: Test target labels.
    
    Returns:
        Dictionary with metrics: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.
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


def train(data_path: str | Path, random_state=42, test_size=0.2) -> dict[str, float]:
    """Run one training cycle, log it to MLflow, return test metrics.
    
    Loads raw data, builds features, splits into train/test sets, trains a
    RandomForestClassifier pipeline, evaluates it, and logs the run to MLflow.
    
    Args:
        data_path: Path to the raw customer CSV file.
        random_state: Random seed for train/test split and model. Default: 42.
        test_size: Fraction of data to use for testing. Default: 0.2.
    
    Returns:
        metrics = evaluate(pipeline, X_test, y_test).
    """
    
    # Load data.
    raw = load_raw(data_path)
    raw = raw[raw["tenure"] > 0]
    logger.info("loaded %d rows from %s", len(raw), data_path)

    # Feature, Prepare data
    X = build_features(raw)
    y = extract_target(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Build, Train, Evaluate pipeline
    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)
    metrics = _evaluate(pipeline, X_test, y_test)

    # MLflow Tracking
    with mlflow.start_run():
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

    # MLflow setup (uri, experiment, runs)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(EXPERIMENT_NAME)

    # MLflow runs and Training process
    metrics = train(args.data)
    
    for name, value in metrics.items():
        print(f"{name:9s} {value:.3f}")


if __name__ == "__main__":
    main()
