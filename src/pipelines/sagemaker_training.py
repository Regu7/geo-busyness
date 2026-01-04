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

    # When using TrainingInput with a name like "config", SageMaker mounts it at:
    # /opt/ml/input/data/config
    # It treats it as a "channel" named "config".
    # Since we passed a single file, it might be inside that folder.
    config_dir = "/opt/ml/input/data/config"

    # Find the config file in the directory
    config_files = [
        f for f in os.listdir(config_dir) if f.endswith(".yaml") or f.endswith(".yml")
    ]
    if not config_files:
        # Fallback: sometimes it might be mounted directly if configured differently,
        # but usually it's a directory. Let's try to be robust.
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

    # Override config loading in train_model by setting environment or modifying the function
    # For now, we'll modify train_model to accept config
    model = train_model(df, config)

    # Save model to SageMaker model directory
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    print(f"Model saved to {model_dir}")
    print("Training completed successfully")


if __name__ == "__main__":
    main()
