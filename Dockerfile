FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
# Install sagemaker-training and sagemaker-inference for custom container support
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir sagemaker-training sagemaker-inference flask gunicorn gevent

# Copy source code
COPY src/ ./src/
COPY src/config/config.yaml ./
COPY main.py ./

# Create directory for logs
RUN mkdir -p /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/app/mlruns
# SageMaker specific environment variables
ENV SAGEMAKER_PROGRAM=src/pipelines/sagemaker_training.py

# Default command (can be overridden by SageMaker)
CMD ["python", "main.py"]