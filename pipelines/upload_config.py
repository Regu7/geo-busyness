#!/usr/bin/env python3
"""
Upload config.yaml to S3 for SageMaker pipeline
"""

import os

import boto3
from dotenv import load_dotenv

load_dotenv()


def upload_config_to_s3():
    bucket = os.environ.get("SAGEMAKER_BUCKET", "geo-busyness")
    config_key = os.environ.get("CONFIG_S3_KEY", "config/config.yaml")
    local_config = os.environ.get("LOCAL_CONFIG_PATH", "src/config/config.yaml")

    if not os.path.exists(local_config):
        raise FileNotFoundError(f"Config file {local_config} not found")

    s3 = boto3.client("s3")
    s3.upload_file(local_config, bucket, config_key)

    s3_uri = f"s3://{bucket}/{config_key}"
    print(f"Config uploaded to: {s3_uri}")
    return s3_uri


if __name__ == "__main__":
    upload_config_to_s3()
