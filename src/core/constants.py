"""
Constants and shared configurations for the geo-busyness pipeline.

Note: Runtime parameters (h3_resolution, k_clusters, test_size, etc.)
are defined in config.yaml - not here. This file contains only
code-level constants that rarely change.
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

# Earth radius in kilometers (for Haversine distance calculation)
EARTH_RADIUS_KM = 6372.8
