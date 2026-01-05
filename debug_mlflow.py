"""
Debug script to test MLflow connectivity to SageMaker MLflow server.
Run this locally before deploying to SageMaker.

Usage:
    python debug_mlflow.py
"""

import os

import boto3
import requests


# Get tracking server info from AWS
def get_mlflow_config():
    """Fetch MLflow tracking server info from SageMaker."""
    mlflow_server_name = "mlflow-geo-bysuness-prod"

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
        print(f"   Is Active: {response.get('IsActive')}")
        print(f"   Role ARN: {response.get('RoleArn')}")
        print(f"   Artifact Store: {response.get('ArtifactStoreUri')}")

        return tracking_uri, tracking_arn, region, response

    except Exception as e:
        print(f"❌ Failed to get MLflow server info: {e}")
        return None, None, None, None


def test_raw_http(tracking_uri, tracking_arn, region):
    """Test raw HTTP request with SigV4 auth."""
    print("\n--- Testing Raw HTTP Request ---")

    try:
        from requests_auth_aws_sigv4 import AWSSigV4

        # Test basic connectivity without auth first
        print(f"\n[Test] Basic HTTPS connectivity to {tracking_uri}...")
        try:
            resp = requests.get(f"{tracking_uri}/health", timeout=10)
            print(
                f"   Health check: {resp.status_code} - {resp.text[:100] if resp.text else 'empty'}"
            )
        except Exception as e:
            print(f"   Health check failed: {e}")

        # Test with SigV4 auth - try different service names
        print(f"\n[Test] SigV4 authenticated request...")

        # The key header that SageMaker MLflow requires
        headers = {
            "x-amzn-sagemaker-mlflow-tracking-server-arn": tracking_arn,
        }

        for service_name in ["sagemaker", "execute-api"]:
            print(f"\n   Trying service: {service_name}")
            try:
                auth = AWSSigV4(service_name, region=region)
                url = f"{tracking_uri}/api/2.0/mlflow/experiments/search"
                resp = requests.post(
                    url, auth=auth, json={}, headers=headers, timeout=30
                )
                if resp.status_code == 200:
                    print(f"✅ Success with service: {service_name}")
                    return True
            except Exception as e:
                print(f"   Error: {e}")

        print(f"❌ All service names failed")
        return False

    except Exception as e:
        print(f"❌ Raw HTTP test failed: {e}")
        return False


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
    tracking_uri, tracking_arn, region, server_info = get_mlflow_config()

    if not tracking_uri:
        print("\n❌ Could not get MLflow server info. Check your AWS credentials.")
        exit(1)

    # Check server status
    if server_info:
        status = server_info.get("TrackingServerStatus")
        is_active = server_info.get("IsActive")
        if status != "Created" or not is_active:
            print(
                f"\n⚠️  Server may not be ready: Status={status}, IsActive={is_active}"
            )

    # Test raw HTTP first (skip if it fails - may need SageMaker role)
    http_ok = test_raw_http(tracking_uri, tracking_arn, region)

    # Try MLflow client anyway - it may use different auth
    print("\n--- Testing MLflow Client (may work even if raw HTTP fails) ---")
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
