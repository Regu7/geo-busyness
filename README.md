# Geo Busyness Prediction Pipeline

This project implements an end-to-end machine learning pipeline for predicting busyness based on geographic data. It leverages AWS SageMaker for processing, training, and deployment, with a CI/CD workflow managed by GitHub Actions.

## Project Structure

- `src/`: Core source code for the application.
  - `core/`: Contains modules for data ingestion, feature engineering, and model training.
  - `pipelines/`: SageMaker-specific scripts for processing and training steps.
  - `config/`: Configuration files.
- `pipelines/`: Orchestration scripts for defining and running the SageMaker pipeline, model approval, and deployment.
- `tests/`: Unit tests for the core logic.
- `.github/workflows/`: CI/CD definitions.
- `Dockerfile`: Definition for the custom container image used in SageMaker steps.

## Architecture

The pipeline consists of the following steps:

1.  **Feature Engineering**: Processes raw data from S3 using a custom container. It calculates distances, generates H3 indexes, and creates embeddings.
2.  **Model Training**: Trains a Random Forest Regressor on the processed features. Hyperparameters are tuned using GridSearchCV.
3.  **Model Registration**: Registers the trained model in the SageMaker Model Registry.

## CI/CD Workflow

The project uses two main workflows:

1.  **CI (`ci.yml`)**: Runs on every push and pull request. It installs dependencies and executes unit tests using `pytest`.
2.  **CD (`sagemaker-pipeline.yml`)**: Triggers after a successful CI run on specific branches.
    - Builds and pushes the Docker image to Amazon ECR.
    - Uploads configuration to S3.
    - Upserts and executes the SageMaker Pipeline.
    - Approves the registered model.
    - Deploys the model to a SageMaker Endpoint.

## Environment Configuration

The pipeline behavior is controlled by environment variables and the branch name:

- **`main` branch**: Deploys to the `prod` environment (e.g., `geo-busyness-prod`).
- **`dev_pipeline` branch**: Deploys to the `dev` environment (e.g., `geo-busyness-dev`).

Key environment variables include:
- `SAGEMAKER_BUCKET`: S3 bucket for data and artifacts.
- `ECR_REPO`: ECR repository name.
- `SAGEMAKER_ROLE_ARN`: IAM role ARN for SageMaker execution.

## Local Development

1.  Install dependencies:
    ```bash
    pip install -r src/requirements.txt
    pip install -e .
    ```

2.  Run tests:
    ```bash
    pytest
    ```

## Deployment

To deploy changes, push to the `main` branch. The GitHub Actions workflow will automatically build the image, run the pipeline, and update the endpoint if the tests pass.
