"""Model batch endpoint: apply the production model to a raw customer extract.

Run from the repository root:

    python -m telco_churn.score [--data PATH] [--out PATH] [--threshold P]

Loads whichever model version the MLflow registry's "production" alias
points at, scores every customer in the extract, and writes a churn-score 
pd.DataFrame (customerID, churn_probability, churn_flag).
"""

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.sklearn import load_model
from sklearn.pipeline import Pipeline

from telco_churn.data import load_raw
from telco_churn.features import build_features

DEFAULT_DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_OUTPUT_PATH = "outputs/churn_scores.csv"
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_URI = "models:/telco-churn@production"
DEFAULT_THRESHOLD = 0.5


def score_frame(model: Pipeline, raw_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Score every customer in a raw extract. Row-preserving.

    Args:
        model: Fitted pipeline.
        raw_df: Raw customer extract.
        threshold: Probability at which a customer is flagged as churn.

    Returns:
        A pd.DataFrame (customerID, churn_probability, churn_flag).
    """
    features = build_features(raw_df)
    proba = model.predict_proba(features)[:, 1]
    return pd.DataFrame(
        {
            "customerID": raw_df["customerID"].to_numpy(),
            "churn_probability": proba.round(4),
            "churn_flag": proba >= threshold,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a customer extract with the production churn model."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the raw customer CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the score table (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Flag customers at or above this probability (default: %(default)s; "
        "untuned -- a business decision, not a statistical one).",
    )
    args = parser.parse_args()

    # MLflow setup (uri)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    
    # Model loading    
    try:
        model = load_model(MODEL_URI)
    except mlflow.MlflowException as exc:
        raise SystemExit(
            f"No model found at {MODEL_URI}. "
            "Run `python -m telco_churn.train`, then assign the 'production' "
            "alias to a model version in the MLflow UI."
        ) from exc
    
    # Model scoring
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores = score_frame(model, load_raw(args.data), args.threshold) # type: ignore
    scores.to_csv(out_path, index=False)
    
    flagged = int(scores["churn_flag"].sum())
    print(f"scored    {len(scores)} customers")
    print(f"flagged   {flagged} ({flagged / len(scores):.1%}) at threshold {args.threshold}")
    print(f"written   {out_path}")


if __name__ == "__main__":
    main()
