#!/usr/bin/env python3
"""
Build and push Docker image to Amazon ECR for SageMaker pipeline
"""
import base64
import os
import subprocess
from pathlib import Path

import boto3


def get_account_id():
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def get_region():
    return boto3.Session().region_name


def create_ecr_repository(repo_name):
    ecr = boto3.client("ecr")
    try:
        ecr.describe_repositories(repositoryNames=[repo_name])
        print(f"Repository {repo_name} already exists")
    except ecr.exceptions.RepositoryNotFoundException:
        ecr.create_repository(repositoryName=repo_name)
        print(f"Created repository {repo_name}")


def build_and_push_image():
    account_id = get_account_id()
    region = get_region()
    repo_name = "geo-busyness"
    image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"

    # Create ECR repository
    create_ecr_repository(repo_name)

    # Get ECR login token
    ecr = boto3.client("ecr")
    token = ecr.get_authorization_token()
    auth_token = token["authorizationData"][0]["authorizationToken"]

    # Decode the base64 token
    decoded_token = base64.b64decode(auth_token).decode("utf-8")
    username, password = decoded_token.split(":")
    registry = token["authorizationData"][0]["proxyEndpoint"]

    # Login to ECR
    subprocess.run(
        ["docker", "login", "--username", username, "--password-stdin", registry],
        input=password.encode("utf-8"),
        check=True,
    )

    # Build image
    project_root = Path(__file__).parent.parent  # Go up one more level to project root
    subprocess.run(
        ["docker", "build", "-t", f"{repo_name}:latest", str(project_root)], check=True
    )

    # Tag image
    subprocess.run(["docker", "tag", f"{repo_name}:latest", image_uri], check=True)

    # Push image
    subprocess.run(["docker", "push", image_uri], check=True)

    print(f"Image pushed to: {image_uri}")
    return image_uri


if __name__ == "__main__":
    print("build starting...")
    build_and_push_image()

