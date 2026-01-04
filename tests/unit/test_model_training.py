from unittest.mock import MagicMock, patch


from src.core.model_training import train_model


@patch("src.core.model_training.joblib.dump")
@patch("src.core.model_training.mlflow")
@patch("src.core.model_training.GridSearchCV")
@patch("src.core.model_training.RandomForestRegressor")
@patch("src.core.model_training.boto3.client")
def test_train_model(
    mock_boto, mock_rf, mock_grid, mock_mlflow, mock_joblib_dump, sample_df, mock_config
):
    # Setup mocks
    mock_grid_instance = MagicMock()
    mock_grid.return_value = mock_grid_instance
    mock_grid_instance.best_score_ = 0.9
    mock_grid_instance.best_estimator_ = MagicMock()

    # Add necessary columns to sample_df for training
    df = sample_df.copy()
    # Add dummy columns expected by train_model
    required_cols = [
        "dist_to_restaurant",
        "Hdist_to_restaurant",
        "avg_Hdist_to_restaurants",
        "date_day_number",
        "restaurant_id",
        "Five_Clusters_embedding",
        "h3_index",
        "date_hour_number",
        "restaurants_per_index",
        "orders_busyness_by_h3_hour",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    # Run training
    train_model(df, config=mock_config)

    # Verify MLflow calls
    mock_mlflow.set_experiment.assert_called_with("geo-busyness-experiment")
    mock_mlflow.start_run.assert_called()
    mock_mlflow.log_param.assert_called()

    # Verify GridSearch
    mock_grid.assert_called()
    mock_grid_instance.fit.assert_called()
    # Verify GridSearch
    mock_grid.assert_called()
    mock_grid_instance.fit.assert_called()
