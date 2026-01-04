import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.core.feature_engineering import (
    calc_dist,
    calc_haversine_dist,
    initiate_centroids,
    centroid_assignation,
    Encoder,
    generate_features
)

def test_calc_dist():
    # 3-4-5 triangle
    dist = calc_dist(0, 0, 3, 4)
    assert dist == 5.0
    
    # Vectorized
    dists = calc_dist([0, 0], [0, 0], [3, 6], [4, 8])
    assert dists == [5.0, 10.0]

def test_calc_haversine_dist():
    # Distance between two points (approximate)
    # London (51.5074, -0.1278) to Paris (48.8566, 2.3522) is ~344 km
    dist = calc_haversine_dist(51.5074, -0.1278, 48.8566, 2.3522)
    assert 340 < dist < 350

def test_initiate_centroids(sample_df):
    k = 2
    centroids = initiate_centroids(k, sample_df)
    assert len(centroids) == k
    assert isinstance(centroids, pd.DataFrame)

def test_centroid_assignation(sample_df):
    # Create dummy centroids
    centroids = sample_df.iloc[:2].copy()
    centroids = centroids.rename(columns={'courier_lat': 'lat', 'courier_lon': 'lon'})
    
    df_result = centroid_assignation(sample_df.copy(), centroids)
    
    assert 'Five_Clusters_embedding' in df_result.columns
    assert 'Five_Clusters_embedding_error' in df_result.columns
    assert len(df_result) == len(sample_df)

def test_encoder(sample_df):
    df_encoded = Encoder(sample_df.copy())
    # 'category' column should be encoded to int
    assert pd.api.types.is_integer_dtype(df_encoded['category']) or pd.api.types.is_float_dtype(df_encoded['category'])

@patch("src.core.feature_engineering.load_config")
def test_generate_features(mock_load_config, mock_config, sample_df):
    mock_load_config.return_value = mock_config
    
    # Create dummy restaurant_ids
    restaurants_ids = {
        "10.05_20.05": {"lat": 10.05, "lon": 20.05, "id": 0},
        "10.15_20.15": {"lat": 10.15, "lon": 20.15, "id": 1},
        "10.25_20.25": {"lat": 10.25, "lon": 20.25, "id": 2},
    }
    
    generate_features(sample_df, restaurants_ids)
    
    assert 'dist_to_restaurant' in sample_df.columns
    assert 'avg_dist_to_restaurants' in sample_df.columns
    assert 'Hdist_to_restaurant' in sample_df.columns
