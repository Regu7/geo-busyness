import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    data = {
        "courier_id": [1, 2, 3, 1],
        "courier_lat": [10.0, 10.1, 10.2, 10.0],
        "courier_lon": [20.0, 20.1, 20.2, 20.0],
        "restaurant_lat": [10.05, 10.15, 10.25, 10.05],
        "restaurant_lon": [20.05, 20.15, 20.25, 20.05],
        "restaurant_id": [0, 1, 2, 0],
        "category": ["A", "B", "A", "B"],
        "courier_location_timestamp": ["2023-01-01T10:00:00Z"] * 4,
        "order_created_timestamp": ["2023-01-01T10:05:00Z"] * 4,
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    return {
        "bucket": "test-bucket",
        "raw_data_path": "raw",
        "file_name": "data.csv",
        "processed_data_path": "processed",
        "h3_resolution": 9,
        "k_clusters": 2,
        "random_seed": 42,
        "random_state": 42,
        "test_size": 0.2,
        "model_params": {"n_estimators": 10},
        "model_artifacts_path": "model_artifacts",
    }
