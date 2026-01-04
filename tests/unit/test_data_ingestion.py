import io
from unittest.mock import MagicMock, patch

import pandas as pd

from src.core.data_ingestion import load_and_process_data, load_config


def test_load_config(mock_config):
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        with patch("yaml.safe_load", return_value=mock_config) as _mock_yaml:
            config = load_config()
            assert config == mock_config
            mock_open.assert_called_once()


@patch("src.core.data_ingestion.validate_dataframe")
@patch("src.core.data_ingestion.load_config")
@patch("src.core.data_ingestion.boto3.client")
def test_load_and_process_data_from_s3(
    mock_boto_client, mock_load_config, mock_validate, mock_config
):
    mock_load_config.return_value = mock_config

    # Mock S3 response
    csv_content = "courier_id,courier_lat,courier_lon,restaurant_lat,restaurant_lon\n1,10.0,20.0,10.05,20.05\n"
    mock_s3 = MagicMock()
    mock_s3.get_object.return_value = {"Body": io.BytesIO(csv_content.encode("utf-8"))}
    mock_boto_client.return_value = mock_s3

    # Mock validation to return the df as-is with no errors
    mock_validate.side_effect = lambda df, _: (df, [])

    df, restaurants_ids = load_and_process_data(df=None)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "restaurant_id" in df.columns
    mock_s3.get_object.assert_called_once()
    mock_s3.put_object.assert_called_once()
    mock_validate.assert_called_once()


@patch("src.core.data_ingestion.load_config")
@patch("src.core.data_ingestion.boto3.client")
def test_load_and_process_data_with_df(
    mock_boto_client, mock_load_config, mock_config, sample_df
):
    mock_load_config.return_value = mock_config
    mock_s3 = MagicMock()
    mock_boto_client.return_value = mock_s3

    df, restaurants_ids = load_and_process_data(df=sample_df.copy())

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert "restaurant_id" in df.columns
    # Should not call get_object if df is provided
    mock_s3.get_object.assert_not_called()
    # Should still save processed data
    mock_s3.put_object.assert_called_once()
    mock_s3.get_object.assert_not_called()
    # Should still save processed data
    mock_s3.put_object.assert_called_once()
