"""Model promotion: point a registry alias at a chosen model version.

Run from the repository root:

    python -m telco_churn.promote                    # list candidates
    python -m telco_churn.promote --version 7
    python -m telco_churn.promote --run-id a1b2c3d4
    python -m telco_churn.promote --best roc_auc

Promotion is deliberately separate from training: training produces
candidates, promotion decides which one serves.
"""

import argparse
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from .config import MODEL_NAME, TRACKING_URI, PRODUCTION_ALIAS

def set_alias(client: MlflowClient, version: str, alias: str = PRODUCTION_ALIAS) -> None:
    """Move ``alias`` to ``version``, reporting what it pointed at before."""
    previous = _current_alias_version(client, alias)
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    print(f"@{alias}: {previous or '(unset)'} -> {version}")


def _current_alias_version(client: MlflowClient, alias: str) -> str | None:
    """Return the version currently assigned to an alias."""
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except MlflowException:
        return None


def _version_for_run(client: MlflowClient, run_id: str) -> str:
    """Return the registered model version created from a run.

    Raises:
        SystemExit: If no registered version was produced by ``run_id``.
    """
    versions = client.search_model_versions(
        f"name='{MODEL_NAME}' and run_id='{run_id}'"
    )
    if not versions:
        raise SystemExit(f"No version of {MODEL_NAME} came from run {run_id}.")
    return versions[0].version


def _best_version(client: MlflowClient, metric: str) -> str:
    """Return the best registered version for a metric.

    The function examines logged metrics on the runs that produced registered
    model versions and returns the version with the largest value for ``metric``.

    Raises:
        SystemExit: If no registered version has a logged metric named ``metric``.
    """
    scored = []
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        run = client.get_run(v.run_id)  # type: ignore
        if metric in run.data.metrics:
            scored.append((run.data.metrics[metric], v.version))
    if not scored:
        raise SystemExit(f"No registered version has a logged '{metric}' metric.")
    value, version = max(scored, key=lambda pair: pair[0])
    print(f"best {metric} = {value:.4f} (version {version})")
    return version


def _list_versions(client: MlflowClient) -> None:
    """List registered versions and their metrics."""
    versions = sorted(
        client.search_model_versions(f"name='{MODEL_NAME}'"),
        key=lambda v: int(v.version),
        reverse=True,
    )
    if not versions:
        raise SystemExit(f"{MODEL_NAME} has no versions. Run the train.py first.")
    print(f"{'ver':>4}  {'run_id':10}  {'roc_auc':>8}  {'f1':>6}  aliases")
    for v in versions:
        m = client.get_run(v.run_id).data.metrics # type: ignore
        print(
            f"{v.version:>4}  {v.run_id[:10]}  {m.get('roc_auc', float('nan')):>8.4f}  " # type: ignore
            f"{m.get('f1', float('nan')):>6.4f}  {','.join(v.aliases) or '-'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Point a registry alias at a chosen model version."
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--version", help="Registered version number to promote.")
    target.add_argument("--run-id", help="Promote the version produced by this run.")
    target.add_argument("--best", metavar="METRIC", help="Promote the highest-scoring version.")
    parser.add_argument(
        "--alias",
        default=PRODUCTION_ALIAS,
        help="Alias to move (default: %(default)s). Use e.g. 'staging' to stage a candidate.",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))
    client = MlflowClient()

    if args.version:
        set_alias(client, args.version, args.alias)
    elif args.run_id:
        set_alias(client, _version_for_run(client, args.run_id), args.alias)
    elif args.best:
        set_alias(client, _best_version(client, args.best), args.alias)
    else:
        _list_versions(client)   # no selector: show the candidates


if __name__ == "__main__":
    main()