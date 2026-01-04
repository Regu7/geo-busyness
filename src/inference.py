import joblib
import os
import json
import logging
import pandas as pd
import numpy as np
from io import StringIO

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

def model_fn(model_dir):
    """
    Deserialize fitted model
    """
    logger.info(f"Loading model from: {model_dir}")
    model_path = os.path.join(model_dir, "model.joblib")
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

def input_fn(request_body, request_content_type):
    """
    Parse input data
    """
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == 'text/csv':
        try:
            # Read CSV string into DataFrame. 
            # Assuming no header for inference requests usually, but let's handle it.
            df = pd.read_csv(StringIO(request_body), header=None)
            logger.info(f"Parsed CSV shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
    elif request_content_type == 'application/json':
        try:
            data = json.loads(request_body)
            # Handle standard SageMaker JSON format {"instances": [...]}
            if isinstance(data, dict) and 'instances' in data:
                df = pd.DataFrame(data['instances'])
            else:
                # Handle raw JSON list or dict
                df = pd.DataFrame(data)
            logger.info(f"Parsed JSON shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            raise
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make prediction
    """
    logger.info("Executing prediction")
    try:
        prediction = model.predict(input_data)
        logger.info(f"Prediction generated: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        # Try converting to numpy if it's a DataFrame and failed (though sklearn handles DF)
        try:
            if isinstance(input_data, pd.DataFrame):
                logger.info("Retrying prediction with numpy array")
                prediction = model.predict(input_data.values)
                return prediction
        except Exception as e2:
            logger.error(f"Retry failed: {e2}")
        raise

def output_fn(prediction, response_content_type):
    """
    Format output
    """
    logger.info(f"Formatting output for content type: {response_content_type}")
    
    if response_content_type == 'application/json':
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        return json.dumps({'predictions': prediction})
    elif response_content_type == 'text/csv':
        return ",".join(map(str, prediction))
    else:
        # Default to JSON
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        return json.dumps({'predictions': prediction})
