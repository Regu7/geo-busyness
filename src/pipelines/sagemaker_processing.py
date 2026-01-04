import argparse
import os
from io import StringIO

import boto3
import joblib
import pandas as pd
import yaml

from src.core.data_ingestion import load_and_process_data
from src.core.feature_engineering import generate_features


def main():
    # SageMaker processing paths
    input_data_path = "/opt/ml/processing/input/raw"
    output_features_path = "/opt/ml/processing/output/features"

    os.makedirs(output_features_path, exist_ok=True)

    # Find the data file in input directory
    data_files = [f for f in os.listdir(input_data_path) if f.endswith(".csv")]
    if not data_files:
        raise FileNotFoundError("No CSV file found in input data")

    file_path = os.path.join(input_data_path, data_files[0])
    df = pd.read_csv(file_path)

    # Process data and generate features
    df_processed, restaurants_ids = load_and_process_data(df=df)
    df_features, artifacts = generate_features(df_processed, restaurants_ids)

    # Add restaurants_ids to artifacts
    artifacts["restaurants_ids"] = restaurants_ids

    # Save processed features
    output_file = os.path.join(output_features_path, "features.csv")
    df_features.to_csv(output_file, index=False)

    # Save artifacts
    artifacts_file = os.path.join(output_features_path, "artifacts.joblib")
    joblib.dump(artifacts, artifacts_file)

    print(f"Features saved to {output_file}")
    print(f"Artifacts saved to {artifacts_file}")


if __name__ == "__main__":
    main()
