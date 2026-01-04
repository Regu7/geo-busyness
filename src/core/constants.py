"""
Constants and shared configurations for the geo-busyness pipeline.
"""

# Feature columns used for model training and inference
FEATURE_COLUMNS = [
    "dist_to_restaurant",
    "Hdist_to_restaurant",
    "avg_Hdist_to_restaurants",
    "date_day_number",
    "restaurant_id",
    "Five_Clusters_embedding",
    "h3_index",
    "date_hour_number",
    "restaurants_per_index",
]

# Target column for prediction
TARGET_COLUMN = "orders_busyness_by_h3_hour"

# Required input columns for data ingestion
REQUIRED_INPUT_COLUMNS = [
    "courier_id",
    "courier_lat",
    "courier_lon",
    "restaurant_lat",
    "restaurant_lon",
    "courier_location_timestamp",
    "order_created_timestamp",
]

# Required columns for inference
REQUIRED_INFERENCE_COLUMNS = [
    "courier_lat",
    "courier_lon",
    "restaurant_lat",
    "restaurant_lon",
    "courier_location_timestamp",
]

# Default H3 resolution for geospatial indexing
DEFAULT_H3_RESOLUTION = 7

# Default number of clusters for restaurant embedding
DEFAULT_K_CLUSTERS = 5

# Default random seed for reproducibility
DEFAULT_RANDOM_SEED = 1

# Default train/test split parameters
DEFAULT_TEST_SIZE = 0.33
DEFAULT_RANDOM_STATE = 42

# Earth radius in kilometers (for Haversine distance calculation)
EARTH_RADIUS_KM = 6372.8
