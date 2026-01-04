import logging
import os
import sys

from flask import Flask, Response, request

# Add /app to path to ensure we can import src
sys.path.append("/app")

from src.model_inference import input_fn, model_fn, output_fn, predict_fn

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model on startup
# SageMaker mounts the model artifacts to /opt/ml/model
MODEL_DIR = "/opt/ml/model"

model = None


def load_model():
    global model
    try:
        if not os.path.exists(MODEL_DIR):
            logger.warning(
                f"Model directory {MODEL_DIR} does not exist. Waiting for model artifacts."
            )
            return None

        model = model_fn(MODEL_DIR)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


# Attempt to load model immediately
load_model()


@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check. SageMaker expects 200 OK if the container is healthy.
    """
    global model
    if model is None:
        model = load_model()

    if model:
        return Response(status=200)
    else:
        # If model fails to load, we should return 500 so SageMaker knows we are not ready
        return Response(status=500)


@app.route("/invocations", methods=["POST"])
def invoke():
    """
    Inference endpoint.
    """
    global model
    if not model:
        model = load_model()
        if not model:
            return Response("Model not loaded", status=500)

    try:
        content_type = request.content_type
        # SageMaker sends data as bytes
        data = request.data.decode("utf-8")

        # 1. Input processing
        input_data = input_fn(data, content_type)

        # 2. Prediction
        prediction = predict_fn(input_data, model)

        # 3. Output formatting
        accept = request.headers.get("Accept", content_type)
        response_body = output_fn(prediction, accept)

        return Response(response_body, status=200, mimetype=accept)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return Response(str(e), status=500)


if __name__ == "__main__":
    # For local testing
    app.run(host="0.0.0.0", port=8080)
