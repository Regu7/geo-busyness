"""
Debug script to test MLflow connectivity to SageMaker MLflow server.
Run this locally before deploying to SageMaker.

Usage:
    python debug_mlflow.py
"""

import os

import boto3


# Get tracking server info from AWS
def get_mlflow_config():
    """Fetch MLflow tracking server info from SageMaker."""
    mlflow_server_name = "mlflow-geo-busyness"

    try:
        session = boto3.Session()
        region = session.region_name or "us-east-1"
        account_id = boto3.client("sts").get_caller_identity()["Account"]

        # Get tracking URL from SageMaker
        sm_client = boto3.client("sagemaker", region_name=region)
        response = sm_client.describe_mlflow_tracking_server(
            TrackingServerName=mlflow_server_name
        )

        tracking_uri = response.get("TrackingServerUrl")
        tracking_arn = f"arn:aws:sagemaker:{region}:{account_id}:mlflow-tracking-server/{mlflow_server_name}"

        print(f"✅ Found MLflow server: {mlflow_server_name}")
        print(f"   Region: {region}")
        print(f"   Account: {account_id}")
        print(f"   Tracking URI: {tracking_uri}")
        print(f"   Tracking ARN: {tracking_arn}")
        print(f"   Server Status: {response.get('TrackingServerStatus')}")

        return tracking_uri, tracking_arn, region

    except Exception as e:
        print(f"❌ Failed to get MLflow server info: {e}")
        return None, None, None


def test_mlflow_connection(tracking_uri, tracking_arn, region):
    """Test MLflow connection with SageMaker auth."""
    import mlflow

    print("\n--- Testing MLflow Connection ---")

    # Set environment variables for SageMaker MLflow auth
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_TRACKING_ARN"] = tracking_arn
    os.environ["MLFLOW_TRACKING_AWS_SIGV4"] = "true"
    os.environ["AWS_DEFAULT_REGION"] = region

    print(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI')}")
    print(f"MLFLOW_TRACKING_ARN: {os.environ.get('MLFLOW_TRACKING_ARN')}")
    print(f"MLFLOW_TRACKING_AWS_SIGV4: {os.environ.get('MLFLOW_TRACKING_AWS_SIGV4')}")
    print(f"AWS_DEFAULT_REGION: {os.environ.get('AWS_DEFAULT_REGION')}")

    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Test 1: List experiments
        print("\n[Test 1] Listing experiments...")
        client = mlflow.MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ Found {len(experiments)} experiments")
        for exp in experiments[:5]:
            print(f"   - {exp.name} (id: {exp.experiment_id})")
    except Exception as e:
        print(f"❌ Failed to list experiments: {e}")
        return False

    try:
        # Test 2: Create/get experiment
        print("\n[Test 2] Creating/getting experiment...")
        experiment_name = "geo-busyness-experiment"
        mlflow.set_experiment(experiment_name)
        print(f"✅ Experiment '{experiment_name}' ready")
    except Exception as e:
        print(f"❌ Failed to set experiment: {e}")
        return False

    try:
        # Test 3: Start a test run
        print("\n[Test 3] Starting test run...")
        with mlflow.start_run(run_name="debug-test-run"):
            mlflow.log_param("test_param", "hello")
            mlflow.log_metric("test_metric", 0.95)
            run_id = mlflow.active_run().info.run_id
            print(f"✅ Test run created: {run_id}")
    except Exception as e:
        print(f"❌ Failed to create run: {e}")
        return False

    print("\n✅ All MLflow tests passed!")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    print("--- Checking Dependencies ---")

    packages = [
        ("mlflow", "mlflow"),
        ("boto3", "boto3"),
        ("requests_auth_aws_sigv4", "requests-auth-aws-sigv4"),
    ]

    all_ok = True
    for module_name, package_name in packages:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - NOT INSTALLED")
            print(f"   Run: pip install {package_name}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    print("=" * 50)
    print("MLflow SageMaker Debug Script")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        print("\n❌ Missing dependencies. Install them first.")
        exit(1)

    print()

    # Get MLflow config from AWS
    tracking_uri, tracking_arn, region = get_mlflow_config()

    if not tracking_uri:
        print("\n❌ Could not get MLflow server info. Check your AWS credentials.")
        exit(1)

    # Test connection
    success = test_mlflow_connection(tracking_uri, tracking_arn, region)

    if success:
        print("\n" + "=" * 50)
        print("✅ MLflow is working! You can now run the pipeline.")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("❌ MLflow connection failed. Check the errors above.")
        print("=" * 50)
        exit(1)
        print("=" * 50)
        exit(1)
