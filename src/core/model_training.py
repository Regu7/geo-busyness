import io
import logging
import os

import boto3
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from src.core.constants import FEATURE_COLUMNS, TARGET_COLUMN

# Load .env file for local development
load_dotenv()

# ---------------------- Logging setup ----------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler("logs/pipeline.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# ---------------------- MLflow Configuration ----------------------
# Environment variables:
#   MLFLOW_TRACKING_URI - The HTTP URL of the MLflow tracking server
#   MLFLOW_TRACKING_ARN - The ARN of the SageMaker MLflow tracking server (required for SageMaker)
#   MLFLOW_TRACKING_AWS_SIGV4 - Set to "true" to enable AWS SigV4 authentication

mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow_tracking_arn = os.environ.get("MLFLOW_TRACKING_ARN")

if mlflow_tracking_uri and mlflow_tracking_arn:
    # SageMaker Managed MLflow - requires AWS SigV4 auth
    os.environ["MLFLOW_TRACKING_AWS_SIGV4"] = "true"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"Using SageMaker MLflow: {mlflow_tracking_uri}")
elif mlflow_tracking_uri:
    # Standard MLflow server (http/https)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    logger.info(f"Using MLflow server: {mlflow_tracking_uri}")
else:
    # Local file-based tracking
    mlflow.set_tracking_uri("file:./mlruns")
    logger.info("Using local MLflow tracking: file:./mlruns")


# ---------------------- Utilities ----------------------
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------- Training ----------------------
def train_model(df: pd.DataFrame, config=None):
    if config is None:
        config = load_config()
    test_size = config["test_size"]
    random_state = config["random_state"]
    params = config["model_params"]
    bucket = config["bucket"]
    model_artifacts_path = config["model_artifacts_path"]

    logger.info("Starting model training")

    # ---------- MLflow Run ----------
    mlflow.set_experiment("geo-busyness-experiment")
    with mlflow.start_run():
        # Log initial config
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        for key, value in params.items():
            mlflow.log_param(f"param_grid_{key}", str(value))

        # ---------- Data ----------
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

        # ---------- Baseline model ----------
        regr = RandomForestRegressor(
            max_depth=4,
            random_state=random_state,
            n_jobs=-1,
        )
        regr.fit(X_train, y_train)

        baseline_score = regr.score(X_test, y_test)
        logger.info(f"Initial model R²: {baseline_score}")

        mlflow.log_metric("initial_r2_score", baseline_score)

        # ---------- Grid Search ----------
        grid_search = GridSearchCV(
            estimator=regr,
            param_grid=params,
            cv=3,
            n_jobs=-1,
            scoring="r2",
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        best_cv_score = grid_search.best_score_
        logger.info(f"Best CV R²: {best_cv_score}")

        mlflow.log_metric("best_cv_r2_score", best_cv_score)

        # ---------- Best model evaluation ----------
        best_model = grid_search.best_estimator_
        test_score = best_model.score(X_test, y_test)
        logger.info(f"Best model test R²: {test_score}")

        mlflow.log_metric("best_test_r2_score", test_score)

        # ---------- Log best hyperparameters ----------
        best_params = grid_search.best_params_
        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)

        # ---------- Save model to S3 ----------
        run_id = mlflow.active_run().info.run_id
        model_key = f"{model_artifacts_path}/{run_id}/random_forest_model.pkl"

        model_buffer = io.BytesIO()
        joblib.dump(best_model, model_buffer)
        model_buffer.seek(0)

        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=model_key,
            Body=model_buffer.getvalue(),
        )

        logger.info(f"Model saved to s3://{bucket}/{model_key}")

        mlflow.log_param("model_s3_path", f"s3://{bucket}/{model_key}")

        # Log model to MLflow
        mlflow.sklearn.log_model(best_model, "random_forest_model")

        return best_model


# ---------------------- Entry point ----------------------
if __name__ == "__main__":
    from data_ingestion import load_and_process_data
    from feature_engineering import generate_features

    logger.info("Starting training pipeline")

    df, restaurants = load_and_process_data()
    df = generate_features(df, restaurants)

    model = train_model(df)

    logger.info("Training pipeline completed successfully")
