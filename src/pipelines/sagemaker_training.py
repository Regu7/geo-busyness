import argparse
import os

import joblib
import pandas as pd
import yaml

try:
    from src.core.model_training import train_model
except ImportError:
    from core.model_training import train_model


def main():
    # SageMaker training environment
    training_data_path = "/opt/ml/input/data/train"
    config_dir = "/opt/ml/input/data/config"

    # Find the config file in the directory
    config_files = [
        f for f in os.listdir(config_dir) if f.endswith(".yaml") or f.endswith(".yml")
    ]
    if not config_files:
        raise FileNotFoundError(
            f"No config file found in {config_dir}. Contents: {os.listdir(config_dir)}"
        )

    config_path = os.path.join(config_dir, config_files[0])
    print(f"Loading config from: {config_path}")

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load training data
    data_files = [f for f in os.listdir(training_data_path) if f.endswith(".csv")]
    if not data_files:
        raise FileNotFoundError("No training data found")

    df = pd.read_csv(os.path.join(training_data_path, data_files[0]))

    # Copy artifacts to model directory so they are packaged with the model
    artifacts_files = [
        f for f in os.listdir(training_data_path) if f.endswith(".joblib")
    ]
    if artifacts_files:
        print(f"Found artifacts: {artifacts_files[0]}")
        import shutil

        src_artifact = os.path.join(training_data_path, artifacts_files[0])
        model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
        dst_artifact = os.path.join(model_dir, "artifacts.joblib")
        shutil.copy(src_artifact, dst_artifact)
        print(f"Copied artifacts to {dst_artifact}")

    model = train_model(df, config)

    # Save model to SageMaker model directory
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    print(f"Model saved to {model_dir}")
    print("Training completed successfully")


if __name__ == "__main__":
    main()
