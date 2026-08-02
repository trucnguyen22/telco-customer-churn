"""Runtime configuration"""
import os

DATA_PATH        = os.environ.get("TELCO_DATA_PATH", "data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
TRACKING_URI     = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME  = os.environ.get("MLFLOW_EXPERIMENT", "telco-churn")
MODEL_NAME       = "telco-churn"
MODEL_URI        = f"models:/{MODEL_NAME}@production"
PRODUCTION_ALIAS = "production"