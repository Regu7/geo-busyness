"""
Integration tests for the geo-busyness pipeline.
Tests the full end-to-end flow of data processing, feature engineering, and model training.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.core.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.core.data_ingestion import load_and_process_data
from src.core.feature_engineering import generate_features
from src.core.model_training import train_model
from src.core.validation import validate_dataframe


@pytest.fixture
def realistic_df():
    """Create a realistic DataFrame for integration testing."""
    np.random.seed(42)
    n_samples = 100

    # Generate realistic coordinates (around a typical city)
    base_lat, base_lon = 40.7128, -74.0060  # NYC area

    data = {
        "courier_id": np.random.randint(1, 20, n_samples),
        "courier_lat": base_lat + np.random.uniform(-0.1, 0.1, n_samples),
        "courier_lon": base_lon + np.random.uniform(-0.1, 0.1, n_samples),
        "restaurant_lat": base_lat + np.random.uniform(-0.05, 0.05, n_samples),
        "restaurant_lon": base_lon + np.random.uniform(-0.05, 0.05, n_samples),
        "courier_location_timestamp": pd.date_range(
            "2024-01-01", periods=n_samples, freq="h"
        ).astype(str),
        "order_created_timestamp": pd.date_range(
            "2024-01-01", periods=n_samples, freq="h"
        ).astype(str),
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "bucket": "test-bucket",
        "raw_data_path": "raw",
        "file_name": "test_data.csv",
        "processed_data_path": "processed",
        "model_artifacts_path": "model_artifacts",
        "h3_resolution": 7,
        "k_clusters": 3,
        "random_seed": 42,
        "random_state": 42,
        "test_size": 0.2,
        "model_params": {
            "max_depth": [2],
            "min_samples_leaf": [10],
            "n_estimators": [10],
        },
    }


class TestDataValidationIntegration:
    """Integration tests for data validation."""

    def test_validate_valid_training_data(self, realistic_df):
        """Test validation passes for valid training data."""
        valid_df, errors = validate_dataframe(realistic_df, "training")

        assert len(valid_df) == len(realistic_df)
        assert len(errors) == 0

    def test_validate_with_invalid_coordinates(self, realistic_df):
        """Test validation catches invalid coordinates."""
        df = realistic_df.copy()
        # Add invalid latitude
        df.loc[0, "courier_lat"] = 200  # Invalid

        valid_df, errors = validate_dataframe(df, "training")

        assert len(errors) == 1
        assert len(valid_df) == len(df) - 1

    def test_validate_inference_data(self, realistic_df):
        """Test validation for inference data (subset of columns)."""
        inference_df = realistic_df[
            [
                "courier_lat",
                "courier_lon",
                "restaurant_lat",
                "restaurant_lon",
                "courier_location_timestamp",
            ]
        ].copy()

        valid_df, errors = validate_dataframe(inference_df, "inference")

        assert len(valid_df) == len(inference_df)
        assert len(errors) == 0


class TestDataIngestionIntegration:
    """Integration tests for data ingestion."""

    @patch("src.core.data_ingestion.load_config")
    @patch("src.core.data_ingestion.boto3.client")
    def test_full_data_ingestion(
        self, mock_boto, mock_config_fn, realistic_df, mock_config
    ):
        """Test full data ingestion flow."""
        mock_config_fn.return_value = mock_config
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        df, restaurants_ids = load_and_process_data(df=realistic_df.copy())

        # Verify data was processed
        assert "restaurant_id" in df.columns
        assert len(df) == len(realistic_df)
        assert isinstance(restaurants_ids, dict)
        assert len(restaurants_ids) > 0

        # Verify S3 upload was called
        mock_s3.put_object.assert_called_once()


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""

    @patch("src.core.feature_engineering.load_config")
    def test_full_feature_engineering(self, mock_config_fn, realistic_df, mock_config):
        """Test full feature engineering flow."""
        mock_config_fn.return_value = mock_config

        # First, create restaurant_ids (simulating data ingestion)
        restaurants_ids = {}
        for a, b in zip(realistic_df.restaurant_lat, realistic_df.restaurant_lon):
            id_key = f"{a}_{b}"
            if id_key not in restaurants_ids:
                restaurants_ids[id_key] = {
                    "lat": a,
                    "lon": b,
                    "id": len(restaurants_ids),
                }

        realistic_df["restaurant_id"] = [
            restaurants_ids[f"{a}_{b}"]["id"]
            for a, b in zip(realistic_df.restaurant_lat, realistic_df.restaurant_lon)
        ]

        # Run feature engineering
        df_features, artifacts = generate_features(realistic_df, restaurants_ids)

        # Verify all feature columns are present
        for col in FEATURE_COLUMNS:
            assert col in df_features.columns, f"Missing column: {col}"

        # Verify target column
        assert TARGET_COLUMN in df_features.columns

        # Verify artifacts
        assert "centroids" in artifacts
        assert "encoders" in artifacts
        assert "restaurants_counts_per_h3_index" in artifacts


class TestModelTrainingIntegration:
    """Integration tests for model training."""

    @patch("src.core.model_training.mlflow")
    @patch("src.core.model_training.boto3.client")
    def test_full_training_pipeline(
        self, mock_boto, mock_mlflow, realistic_df, mock_config
    ):
        """Test full model training flow."""
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        # Prepare data with all required columns
        np.random.seed(42)
        df = realistic_df.copy()

        # Add feature columns
        df["dist_to_restaurant"] = np.random.uniform(0, 1, len(df))
        df["Hdist_to_restaurant"] = np.random.uniform(0, 10, len(df))
        df["avg_Hdist_to_restaurants"] = np.random.uniform(0, 5, len(df))
        df["date_day_number"] = np.random.randint(1, 365, len(df))
        df["restaurant_id"] = np.random.randint(0, 10, len(df))
        df["Five_Clusters_embedding"] = np.random.randint(0, 3, len(df))
        df["h3_index"] = np.random.randint(0, 50, len(df))
        df["date_hour_number"] = np.random.randint(0, 24, len(df))
        df["restaurants_per_index"] = np.random.randint(1, 5, len(df))
        df["orders_busyness_by_h3_hour"] = np.random.randint(1, 100, len(df))

        # Train model
        model = train_model(df, config=mock_config)

        # Verify model was trained
        assert model is not None
        assert hasattr(model, "predict")

        # Verify MLflow calls
        mock_mlflow.set_experiment.assert_called()
        mock_mlflow.start_run.assert_called()

        # Verify S3 upload
        mock_s3.put_object.assert_called()


class TestEndToEndPipeline:
    """End-to-end integration tests for the complete pipeline."""

    @patch("src.core.model_training.mlflow")
    @patch("src.core.model_training.boto3.client")
    @patch("src.core.data_ingestion.boto3.client")
    @patch("src.core.feature_engineering.load_config")
    @patch("src.core.data_ingestion.load_config")
    def test_complete_pipeline(
        self,
        mock_ingestion_config,
        mock_feature_config,
        mock_ingestion_boto,
        mock_training_boto,
        mock_mlflow,
        realistic_df,
        mock_config,
    ):
        """Test complete end-to-end pipeline execution."""
        # Setup mocks
        mock_ingestion_config.return_value = mock_config
        mock_feature_config.return_value = mock_config

        mock_s3_ingestion = MagicMock()
        mock_s3_training = MagicMock()
        mock_ingestion_boto.return_value = mock_s3_ingestion
        mock_training_boto.return_value = mock_s3_training

        # Step 1: Data Ingestion
        df, restaurants_ids = load_and_process_data(df=realistic_df.copy())
        assert "restaurant_id" in df.columns

        # Step 2: Feature Engineering
        df_features, artifacts = generate_features(df, restaurants_ids)
        for col in FEATURE_COLUMNS:
            assert col in df_features.columns

        # Step 3: Model Training
        model = train_model(df_features, config=mock_config)
        assert model is not None

        # Step 4: Verify model can make predictions
        X_test = df_features[FEATURE_COLUMNS].head(5)
        predictions = model.predict(X_test)
        assert len(predictions) == 5
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)


class TestInferencePipelineIntegration:
    """Integration tests for the inference pipeline."""

    @patch("src.core.feature_engineering.load_config")
    @patch("src.core.data_ingestion.load_config")
    @patch("src.core.data_ingestion.boto3.client")
    def test_inference_with_saved_artifacts(
        self,
        mock_boto,
        mock_ingestion_config,
        mock_feature_config,
        realistic_df,
        mock_config,
    ):
        """Test inference using saved artifacts."""
        mock_ingestion_config.return_value = mock_config
        mock_feature_config.return_value = mock_config
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        # Prepare training data
        df, restaurants_ids = load_and_process_data(df=realistic_df.copy())
        df_features, artifacts = generate_features(df, restaurants_ids)

        # Save artifacts to temp file
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                temp_path = f.name
                artifacts["restaurants_ids"] = restaurants_ids
                joblib.dump(artifacts, temp_path)

            # Load artifacts back (outside the with block to release file handle)
            loaded_artifacts = joblib.load(temp_path)

            # Verify artifacts can be reused
            assert "centroids" in loaded_artifacts
            assert "encoders" in loaded_artifacts
            assert "restaurants_ids" in loaded_artifacts
        finally:
            # Clean up
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
