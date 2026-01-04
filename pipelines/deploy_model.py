import json
import os
import time

import boto3
import sagemaker


def get_latest_approved_model_package(sm_client, model_package_group_name):
    """
    Finds the latest approved model package in the given group.
    """
    # List model packages in the group
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        ModelApprovalStatus="Approved",
    )

    model_packages = response.get("ModelPackageSummaryList", [])
    if not model_packages:
        print(f"No approved model packages found in group: {model_package_group_name}")
        return None

    # Return the latest one
    return model_packages[0]


def _create_model_and_endpoint_config(
    sm_client,
    *,
    role_arn: str,
    model_package_arn: str,
    endpoint_name: str,
    instance_type: str,
):
    timestamp = int(time.time())
    model_name = f"geo-busyness-model-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"

    print(f"Creating Model: {model_name}")
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        Containers=[{"ModelPackageName": model_package_arn}],
    )

    print(f"Creating Endpoint Config: {endpoint_config_name}")
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )

    return model_name, endpoint_config_name


def deploy_model(
    sm_client, model_package_arn, role_arn, endpoint_name, instance_type="ml.t2.medium"
):
    """
    Deploys the given model package to an endpoint using low-level boto3.
    """
    _, endpoint_config_name = _create_model_and_endpoint_config(
        sm_client,
        role_arn=role_arn,
        model_package_arn=model_package_arn,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
    )

    # 3. Create Endpoint (only for first-time creation)
    print(f"Creating Endpoint: {endpoint_name}")
    sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    print(f"Waiting for endpoint {endpoint_name} to be InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    print(f"Model deployed successfully to endpoint: {endpoint_name}")


def update_endpoint(
    sm_client,
    *,
    role_arn: str,
    model_package_arn: str,
    endpoint_name: str,
    instance_type: str,
):
    _, endpoint_config_name = _create_model_and_endpoint_config(
        sm_client,
        role_arn=role_arn,
        model_package_arn=model_package_arn,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
    )

    print(f"Updating Endpoint: {endpoint_name}")
    sm_client.update_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    print(f"Triggered update for endpoint {endpoint_name}...")


def create_api_gateway_role(iam_client, role_name):
    """
    Creates an IAM role that allows API Gateway to invoke SageMaker endpoints.
    """
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {"Service": "apigateway.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        role = iam_client.create_role(
            RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume_role_policy)
        )
        print(f"Created IAM role: {role_name}")
    except iam_client.exceptions.EntityAlreadyExistsException:
        print(f"IAM role {role_name} already exists. Using existing role.")
        role = iam_client.get_role(RoleName=role_name)
    except Exception as e:
        print(f"Error creating IAM role: {e}")
        return None

    policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    try:
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
    except Exception as e:
        print(f"Warning: Could not attach policy to role: {e}")

    return role["Role"]["Arn"]


def create_api_gateway(region, endpoint_name, role_arn):
    """
    Creates an API Gateway REST API to proxy requests to the SageMaker endpoint.
    """
    apigateway = boto3.client("apigateway", region_name=region)
    api_name = f"{endpoint_name}-api"

    print(f"Setting up API Gateway: {api_name}")

    # 1. Create or Get API
    apis = apigateway.get_rest_apis()
    api_id = None
    for item in apis.get("items", []):
        if item["name"] == api_name:
            api_id = item["id"]
            break

    if not api_id:
        api = apigateway.create_rest_api(
            name=api_name, description="API for Geo Busyness Model"
        )
        api_id = api["id"]
        print(f"Created REST API: {api_id}")
    else:
        print(f"Found existing REST API: {api_id}")

    # 2. Get Root Resource
    resources = apigateway.get_resources(restApiId=api_id)
    root_id = resources["items"][0]["id"]

    # 3. Create Resource /predict
    resource_id = None
    for item in resources["items"]:
        if item.get("pathPart") == "predict":
            resource_id = item["id"]
            break

    if not resource_id:
        resource = apigateway.create_resource(
            restApiId=api_id, parentId=root_id, pathPart="predict"
        )
        resource_id = resource["id"]
        print("Created resource /predict")

    # 4. Create Method POST
    try:
        apigateway.put_method(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod="POST",
            authorizationType="NONE",
        )
    except apigateway.exceptions.ConflictException:
        pass  # Method already exists

    # 5. Integration
    # URI format for SageMaker Runtime
    uri = f"arn:aws:apigateway:{region}:runtime.sagemaker:path//endpoints/{endpoint_name}/invocations"

    apigateway.put_integration(
        restApiId=api_id,
        resourceId=resource_id,
        httpMethod="POST",
        type="AWS",
        integrationHttpMethod="POST",
        uri=uri,
        credentials=role_arn,
    )

    # 6. Method Response
    try:
        apigateway.put_method_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod="POST",
            statusCode="200",
        )
    except apigateway.exceptions.ConflictException:
        pass

    # 7. Integration Response
    try:
        apigateway.put_integration_response(
            restApiId=api_id,
            resourceId=resource_id,
            httpMethod="POST",
            statusCode="200",
        )
    except apigateway.exceptions.ConflictException:
        pass

    # 8. Deploy
    apigateway.create_deployment(restApiId=api_id, stageName="prod")

    url = f"https://{api_id}.execute-api.{region}.amazonaws.com/prod/predict"
    print(f"API Gateway deployed: {url}")
    return url


def main():
    # Configuration
    model_package_group_name = os.environ.get(
        "MODEL_PACKAGE_GROUP_NAME", "geo-busyness-model-group"
    )
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "geo-busyness-endpoint")
    instance_type = os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.t2.medium")
    create_apigw = os.environ.get("CREATE_APIGW", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    region = boto3.Session().region_name

    sm_client = boto3.client("sagemaker", region_name=region)
    iam_client = boto3.client("iam", region_name=region)

    # Get execution role
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            print("Could not get execution role from environment.")
            role = "arn:aws:iam::872869206989:role/service-role/AmazonSageMakerAdminIAMExecutionRole"

    # 1. Find latest approved model
    latest_model = get_latest_approved_model_package(
        sm_client, model_package_group_name
    )

    if not latest_model:
        print("Skipping deployment.")
        return

    model_package_arn = latest_model["ModelPackageArn"]
    print(f"Found latest approved model: {model_package_arn}")

    # 2. Check if endpoint exists
    endpoint_status = None
    current_model_arn = None
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        endpoint_status = response["EndpointStatus"]

        # Get current model ARN
        config_name = response["EndpointConfigName"]
        config_response = sm_client.describe_endpoint_config(
            EndpointConfigName=config_name
        )
        # Assuming single variant
        model_name = config_response["ProductionVariants"][0]["ModelName"]
        model_response = sm_client.describe_model(ModelName=model_name)
        # Check if it's a Model Package based model
        if (
            "Containers" in model_response
            and "ModelPackageName" in model_response["Containers"][0]
        ):
            current_model_arn = model_response["Containers"][0]["ModelPackageName"]
        elif (
            "PrimaryContainer" in model_response
            and "ModelPackageName" in model_response["PrimaryContainer"]
        ):
            current_model_arn = model_response["PrimaryContainer"]["ModelPackageName"]

        print(f"Endpoint {endpoint_name} already exists. Status: {endpoint_status}")
        print(f"Current Model ARN: {current_model_arn}")
    except Exception as e:
        # Catch generic exception to be safe, check message
        if "Could not find endpoint" in str(e):
            endpoint_exists = False
            print(f"Endpoint {endpoint_name} does not exist. Creating...")
        else:
            raise e

    # 3. Deploy (Create or Update)
    if endpoint_exists:
        # Check if update is needed
        if current_model_arn == model_package_arn:
            print(
                f"Endpoint is already running the latest model ({model_package_arn}). Skipping update."
            )
        else:
            # Handle Failed state
            if endpoint_status == "Failed":
                print(f"Endpoint is in Failed state. Deleting {endpoint_name}...")
                sm_client.delete_endpoint(EndpointName=endpoint_name)
                print("Waiting for endpoint deletion...")
                sm_client.get_waiter("endpoint_deleted").wait(
                    EndpointName=endpoint_name
                )
                print("Re-creating endpoint...")
                deploy_model(
                    sm_client,
                    model_package_arn,
                    role,
                    endpoint_name,
                    instance_type=instance_type,
                )

            # Wait if endpoint is busy
            elif endpoint_status in [
                "Creating",
                "Updating",
                "SystemUpdating",
                "RollingBack",
            ]:
                print(
                    f"Endpoint is {endpoint_status}. Waiting for it to be InService..."
                )
                waiter = sm_client.get_waiter("endpoint_in_service")
                waiter.wait(EndpointName=endpoint_name)
                print("Endpoint is now InService.")

                # Proceed with update after wait
                update_endpoint(
                    sm_client,
                    role_arn=role,
                    model_package_arn=model_package_arn,
                    endpoint_name=endpoint_name,
                    instance_type=instance_type,
                )

            else:
                update_endpoint(
                    sm_client,
                    role_arn=role,
                    model_package_arn=model_package_arn,
                    endpoint_name=endpoint_name,
                    instance_type=instance_type,
                )

    else:
        deploy_model(
            sm_client,
            model_package_arn,
            role,
            endpoint_name,
            instance_type=instance_type,
        )

    # 4. Create API Gateway
    if create_apigw:
        print("\n--- Setting up API Gateway ---")
        apigw_role_arn = create_api_gateway_role(
            iam_client, "APIGatewaySageMakerInvokeRole"
        )

        if apigw_role_arn:
            # Give IAM a moment to propagate
            time.sleep(10)
            api_url = create_api_gateway(region, endpoint_name, apigw_role_arn)
            print(f"\nSuccess! You can invoke your model at:")
            print(f"POST {api_url}")
            print(f"Body: CSV data (e.g., '10,20,30')")


if __name__ == "__main__":
    main()
