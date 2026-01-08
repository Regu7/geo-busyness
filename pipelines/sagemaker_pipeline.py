import argparse
import os
import time

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput, Processor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingInput, TrainingStep

load_dotenv()

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
region = boto3.Session().region_name

role = os.environ.get("SAGEMAKER_ROLE_ARN")
if not role:
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        role = "arn:aws:iam::872869206989:role/service-role/AmazonSageMaker-ExecutionRole-20260102T193388"

pipeline_session = PipelineSession()

BUCKET = os.environ.get("SAGEMAKER_BUCKET", "geo-busyness")
ECR_REPO = os.environ.get("ECR_REPO", "geo-busyness")
ECR_IMAGE_TAG = os.environ.get("ECR_IMAGE_TAG", "latest")
ECR_IMAGE_URI = (
    f"{boto3.client('sts').get_caller_identity()['Account']}.dkr.ecr.{region}.amazonaws.com/"
    f"{ECR_REPO}:{ECR_IMAGE_TAG}"
)
PIPELINE_NAME = os.environ.get("SAGEMAKER_PIPELINE_NAME", "geo-busyness-pipeline")
MODEL_PACKAGE_GROUP_NAME_VAR = os.environ.get(
    "SAGEMAKER_MODEL_PACKAGE_GROUP_NAME", "geo-busyness-model-group"
)
CONFIG_S3_URI = os.environ.get("CONFIG_S3_URI", f"s3://{BUCKET}/config/config.yaml")


# MLflow configuration from environment variables
# MLFLOW_TRACKING_URI: HTTP URL of the tracking server
# MLFLOW_TRACKING_ARN: ARN of SageMaker MLflow tracking server (required for SageMaker auth)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_ARN = os.environ.get("MLFLOW_TRACKING_ARN")

if MLFLOW_TRACKING_URI:
    print(f"Using MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    if MLFLOW_TRACKING_ARN:
        print(f"Using MLflow Tracking ARN: {MLFLOW_TRACKING_ARN}")
else:
    print("No MLflow Tracking URI found. Training will use local tracking.")

# ------------------------------------------------------------------
# Step 1: Feature Engineering
# ------------------------------------------------------------------
feature_processor = Processor(
    image_uri=ECR_IMAGE_URI,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    entrypoint=["python", "/app/src/pipelines/sagemaker_processing.py"],
    sagemaker_session=pipeline_session,
)

feature_step = ProcessingStep(
    name="FeatureEngineering",
    processor=feature_processor,
    inputs=[
        ProcessingInput(
            source=f"s3://{BUCKET}/data/raw/",
            destination="/opt/ml/processing/input/raw",
        ),
        ProcessingInput(source=CONFIG_S3_URI, destination="/opt/ml/processing/config"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="features",
            source="/opt/ml/processing/output/features",
            destination=f"s3://{BUCKET}/data/features/",
        )
    ],
)

# ------------------------------------------------------------------
# Step 2: Training
# ------------------------------------------------------------------

estimator = Estimator(
    image_uri=ECR_IMAGE_URI,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{BUCKET}/model_artifacts/",
    sagemaker_session=pipeline_session,
    environment={
        "SAGEMAKER_PROGRAM": "/app/src/pipelines/sagemaker_training.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/app",
        "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI or "file:/app/mlruns",
        "MLFLOW_TRACKING_ARN": MLFLOW_TRACKING_ARN or "",
        "AWS_DEFAULT_REGION": region,
    },
)

training_step = TrainingStep(
    name="ModelTraining",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=feature_step.properties.ProcessingOutputConfig.Outputs[
                "features"
            ].S3Output.S3Uri
        ),
        "config": TrainingInput(s3_data=CONFIG_S3_URI, content_type="application/yaml"),
    },
)

# ------------------------------------------------------------------
# Step 3: Evaluation
# ------------------------------------------------------------------
script_eval = Processor(
    image_uri=ECR_IMAGE_URI,
    role=role,
    instance_count=1,
    instance_type="ml.t3.medium",
    entrypoint=["python", "/app/src/pipelines/evaluate.py"],
    sagemaker_session=pipeline_session,
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=script_eval,
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
        ),
        ProcessingInput(
            source=feature_step.properties.ProcessingOutputConfig.Outputs[
                "features"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
        ),
        ProcessingInput(source=CONFIG_S3_URI, destination="/opt/ml/processing/config"),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation",
            destination=f"s3://{BUCKET}/evaluation/",
        ),
    ],
    property_files=[evaluation_report],
)

# ------------------------------------------------------------------
# Step 4: Register Model (Conditional)
# ------------------------------------------------------------------

model = Model(
    image_uri=ECR_IMAGE_URI,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session,
    env={
        "SAGEMAKER_PROGRAM": "src/inference/model_inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/app",
    },
)

register_step = ModelStep(
    name="RegisterModel",
    step_args=model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP_NAME_VAR,
    ),
)

# Condition step for evaluating model quality and branching execution
cond_lte = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=evaluation_step.name,
        property_file=evaluation_report,
        json_path="regression_metrics.r2_score.value",
    ),
    right=0.6,  # Threshold for R2 score
)

condition_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[cond_lte],
    if_steps=[register_step],
    else_steps=[],
)

# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------
pipeline = Pipeline(
    name=PIPELINE_NAME,
    steps=[feature_step, training_step, evaluation_step, condition_step],
    sagemaker_session=pipeline_session,
)


def _wait_for_pipeline_execution(
    execution_arn: str, poll_seconds: int, timeout_minutes: int
) -> None:
    sm = boto3.client("sagemaker")
    deadline = time.time() + timeout_minutes * 60
    terminal_statuses = {"Succeeded", "Failed", "Stopped"}

    while True:
        desc = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
        status = desc.get("PipelineExecutionStatus")
        if status in terminal_statuses:
            print(f"Pipeline execution finished. status={status} arn={execution_arn}")
            if status != "Succeeded":
                raise RuntimeError(
                    f"Pipeline execution did not succeed. status={status} arn={execution_arn}"
                )
            return

        if time.time() > deadline:
            raise TimeoutError(
                f"Timed out waiting for pipeline execution. arn={execution_arn}"
            )

        time.sleep(poll_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wait", action="store_true", help="Wait for pipeline execution to finish"
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-minutes", type=int, default=180)
    args = parser.parse_args()

    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print(f"Pipeline started: {execution.arn}")

    if args.wait:
        _wait_for_pipeline_execution(
            execution_arn=execution.arn,
            poll_seconds=args.poll_seconds,
            timeout_minutes=args.timeout_minutes,
        )
