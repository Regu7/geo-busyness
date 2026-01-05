import json
import logging
import os
from io import StringIO

import h3
import joblib
import numpy as np
import pandas as pd

from src.core.constants import FEATURE_COLUMNS
from src.core.feature_engineering import calc_dist, calc_haversine_dist
from src.core.validation import validate_dataframe

# Fallback H3 resolution if not in artifacts (matches config.yaml default)
_FALLBACK_H3_RESOLUTION = 7

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


def model_fn(model_dir):
    """
    Deserialize fitted model and artifacts
    """
    logger.info(f"Loading model from: {model_dir}")
    model_path = os.path.join(model_dir, "model.joblib")
    artifacts_path = os.path.join(model_dir, "artifacts.joblib")

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully.")

        artifacts = {}
        if os.path.exists(artifacts_path):
            logger.info(f"Loading artifacts from: {artifacts_path}")
            artifacts = joblib.load(artifacts_path)
        else:
            logger.warning(
                "Artifacts file not found. Stateful features will use defaults."
            )

        return {"model": model, "artifacts": artifacts}
    except Exception as e:
        logger.error(f"Failed to load model/artifacts: {str(e)}")
        raise


def transform_data(df, artifacts):
    """
    Apply feature engineering to the input dataframe using saved artifacts.
    """
    logger.info("Starting feature engineering on input data")

    # Ensure required columns exist
    required_cols = [
        "courier_lat",
        "courier_lon",
        "restaurant_lat",
        "restaurant_lon",
        "courier_location_timestamp",
    ]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input data missing required columns: {required_cols}")

    df["dist_to_restaurant"] = calc_dist(
        df.courier_lat, df.courier_lon, df.restaurant_lat, df.restaurant_lon
    )

    df["Hdist_to_restaurant"] = calc_haversine_dist(
        df.courier_lat.tolist(),
        df.courier_lon.tolist(),
        df.restaurant_lat.tolist(),
        df.restaurant_lon.tolist(),
    )

    df["courier_location_timestamp"] = pd.to_datetime(
        df["courier_location_timestamp"], format="ISO8601"
    )
    df["date_day_number"] = df.courier_location_timestamp.dt.day_of_year
    df["date_hour_number"] = df.courier_location_timestamp.dt.hour

    # Use resolution from artifacts (saved during training) or fallback
    resolution = artifacts.get("h3_resolution", _FALLBACK_H3_RESOLUTION)
    df["h3_index"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for (lat, lon) in zip(df.courier_lat, df.courier_lon)
    ]

    restaurants_ids = artifacts.get("restaurants_ids", {})
    centroids = artifacts.get("centroids", pd.DataFrame())
    restaurants_counts_per_h3_index = artifacts.get(
        "restaurants_counts_per_h3_index", {}
    )
    encoders = artifacts.get("encoders", {})

    if restaurants_ids:

        def avg_Hdist_to_restaurants(courier_lat, courier_lon):
            return np.mean(
                [
                    calc_haversine_dist(v["lat"], v["lon"], courier_lat, courier_lon)
                    for v in restaurants_ids.values()
                ]
            )

        df["avg_Hdist_to_restaurants"] = [
            avg_Hdist_to_restaurants(lat, lon)
            for lat, lon in zip(df.courier_lat, df.courier_lon)
        ]
    else:
        df["avg_Hdist_to_restaurants"] = 0.0

    if restaurants_ids:
        df["restaurant_id"] = [
            restaurants_ids.get(f"{a}_{b}", {"id": 0})["id"]
            for a, b in zip(df.restaurant_lat, df.restaurant_lon)
        ]
    else:
        df["restaurant_id"] = 0

    if not centroids.empty:
        assignation = []
        centroids_list = [c for i, c in centroids.iterrows()]
        for i, obs in df.iterrows():
            all_errors = [
                calc_dist(
                    centroid["lat"],
                    centroid["lon"],
                    obs["courier_lat"],
                    obs["courier_lon"],
                )
                for centroid in centroids_list
            ]

            nearest_centroid = np.argmin(all_errors)
            assignation.append(nearest_centroid)
        df["Five_Clusters_embedding"] = assignation
    else:
        df["Five_Clusters_embedding"] = 0

    if restaurants_counts_per_h3_index:
        df["restaurants_per_index"] = [
            restaurants_counts_per_h3_index.get(h, 0) for h in df.h3_index
        ]
    else:
        df["restaurants_per_index"] = 0

    if "h3_index" in encoders:
        le = encoders["h3_index"]
        known_classes = set(le.classes_)
        df["h3_index"] = df["h3_index"].apply(
            lambda x: x if x in known_classes else le.classes_[0]
        )
        df["h3_index"] = le.transform(df["h3_index"])
    else:
        df["h3_index"] = 0

    return df[FEATURE_COLUMNS]


def input_fn(request_body, request_content_type):
    """
    Parse input data
    """
    logger.info(f"Received request with content type: {request_content_type}")

    if request_content_type == "text/csv":
        try:
            df = pd.read_csv(StringIO(request_body))
            logger.info(f"Parsed CSV shape: {df.shape}")
            # Validate input data
            df, errors = validate_dataframe(df, "inference")
            if errors:
                logger.warning(f"Validation dropped {len(errors)} invalid rows")
            if df.empty:
                raise ValueError("No valid data after validation")
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            raise
    elif request_content_type == "application/json":
        try:
            data = json.loads(request_body)
            if isinstance(data, dict) and "instances" in data:
                df = pd.DataFrame(data["instances"])
            else:
                df = pd.DataFrame(data)
            logger.info(f"Parsed JSON shape: {df.shape}")
            # Validate input data
            df, errors = validate_dataframe(df, "inference")
            if errors:
                logger.warning(f"Validation dropped {len(errors)} invalid rows")
            if df.empty:
                raise ValueError("No valid data after validation")
            return df
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            raise
    else:
        logger.error(f"Unsupported content type: {request_content_type}")
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_and_artifacts):
    """
    Make prediction
    """
    logger.info("Executing prediction")
    try:
        model = model_and_artifacts["model"]
        artifacts = model_and_artifacts["artifacts"]

        # Apply feature engineering
        features = transform_data(input_data, artifacts)

        prediction = model.predict(features)
        logger.info(f"Prediction generated: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def output_fn(prediction, response_content_type):
    """
    Format output
    """
    logger.info(f"Formatting output for content type: {response_content_type}")

    if response_content_type == "application/json":
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        return json.dumps({"predictions": prediction})
    elif response_content_type == "text/csv":
        return ",".join(map(str, prediction))
    else:
        # Default to JSON
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
        return json.dumps({"predictions": prediction})
