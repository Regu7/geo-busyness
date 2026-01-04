import io
import logging
import os

import boto3
import numpy as np
import pandas as pd
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# File handler for logs
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/pipeline.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_and_process_data(df=None):
    config = load_config()
    bucket = config["bucket"]
    s3 = boto3.client("s3")
    if df is None:
        key = f"{config['raw_data_path']}/{config['file_name']}"
        obj = s3.get_object(Bucket=bucket, Key=key)
        logger.info(f"Loading data from S3: s3://{bucket}/{key}")
        df = pd.read_csv(io.BytesIO(obj["Body"].read()))
    else:
        logger.info("Using provided dataframe")

    logger.info(f"Loaded dataset with {len(df)} rows")
    df.dropna(axis=0, inplace=True)
    logger.info(f"After dropping NAs: {len(df)} rows")

    # Process unique restaurants
    restaurants_ids = {}
    for a, b in zip(df.restaurant_lat, df.restaurant_lon):
        id_key = f"{a}_{b}"
        restaurants_ids[id_key] = {"lat": a, "lon": b}

    for i, key in enumerate(restaurants_ids.keys()):
        restaurants_ids[key]["id"] = i

    # Labeling of restaurants
    df["restaurant_id"] = [
        restaurants_ids[f"{a}_{b}"]["id"]
        for a, b in zip(df.restaurant_lat, df.restaurant_lon)
    ]

    # Save processed data to S3
    processed_key = f"{config['processed_data_path']}/processed_dataset.csv"
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=processed_key, Body=csv_buffer.getvalue())
    logger.info(f"Processed data saved to S3: s3://{bucket}/{processed_key}")

    return df, restaurants_ids


if __name__ == "__main__":
    df, restaurants = load_and_process_data()
    logger.info(f"Number of unique couriers: {len(df.courier_id.unique())}")
    logger.info(f"Number of unique restaurants: {len(restaurants)}")
