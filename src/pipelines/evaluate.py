import json
import logging
import os
import tarfile

import joblib
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.core.constants import FEATURE_COLUMNS, TARGET_COLUMN

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    logger.info("Starting evaluation...")

    # Paths
    model_path = "/opt/ml/processing/model/model.tar.gz"
    input_data_path = "/opt/ml/processing/input/features.csv"
    config_path = "/opt/ml/processing/config/config.yaml"
    output_path = "/opt/ml/processing/evaluation"

    os.makedirs(output_path, exist_ok=True)

    # Load config
    config = load_config(config_path)
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)

    # Load data
    logger.info(f"Loading data from {input_data_path}")
    df = pd.read_csv(input_data_path)

    # Prepare data using shared constants
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Split data
    logger.info("Splitting data...")
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Load model
    logger.info(f"Extracting model from {model_path}")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    logger.info("Loading model.joblib")
    model = joblib.load("model.joblib")

    # Evaluate
    logger.info("Predicting on test set...")
    predictions = model.predict(X_test)
    score = r2_score(y_test, predictions)
    logger.info(f"R2 Score: {score}")

    # Save evaluation report
    report_dict = {
        "regression_metrics": {
            "r2_score": {"value": score}
        },
    }

    output_file = os.path.join(output_path, "evaluation.json")
    with open(output_file, "w") as f:
        json.dump(report_dict, f)

    logger.info(f"Evaluation report saved to {output_file}")


if __name__ == "__main__":
    main()
