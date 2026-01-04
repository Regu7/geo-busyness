import os
import sys

import boto3
from dotenv import load_dotenv

load_dotenv()


def approve_latest_model_package(model_package_group_name):
    sm_client = boto3.client("sagemaker")

    # List model packages
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
    )

    model_packages = response.get("ModelPackageSummaryList", [])
    if not model_packages:
        print(f"No model packages found in group: {model_package_group_name}")
        return None

    latest_pkg = model_packages[0]
    pkg_arn = latest_pkg["ModelPackageArn"]
    status = latest_pkg["ModelApprovalStatus"]

    print(f"Latest Model Package: {pkg_arn}")
    print(f"Current Status: {status}")

    if status != "Approved":
        print(f"Approving model package {pkg_arn}...")
        sm_client.update_model_package(
            ModelPackageArn=pkg_arn, ModelApprovalStatus="Approved"
        )
        print("Model package approved successfully.")
    else:
        print("Model package is already approved.")


if __name__ == "__main__":
    group_name = os.environ.get("MODEL_PACKAGE_GROUP_NAME", "geo-busyness-model-group")
    approve_latest_model_package(group_name)
