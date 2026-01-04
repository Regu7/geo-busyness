"""
Data validation schemas using Pydantic for the geo-busyness pipeline.
"""

import logging
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.core.constants import REQUIRED_INFERENCE_COLUMNS, REQUIRED_INPUT_COLUMNS

logger = logging.getLogger(__name__)


class CourierLocationRecord(BaseModel):
    """Schema for a single courier location record."""

    model_config = ConfigDict(strict=False)  # Allow type coercion

    courier_id: str | int = Field(..., description="Unique identifier for the courier")
    courier_lat: float = Field(..., ge=-90, le=90, description="Courier latitude")
    courier_lon: float = Field(..., ge=-180, le=180, description="Courier longitude")
    restaurant_lat: float = Field(..., ge=-90, le=90, description="Restaurant latitude")
    restaurant_lon: float = Field(
        ..., ge=-180, le=180, description="Restaurant longitude"
    )
    courier_location_timestamp: datetime = Field(
        ..., description="Timestamp of courier location"
    )
    order_created_timestamp: datetime = Field(
        ..., description="Timestamp when order was created"
    )

    @field_validator("courier_lat", "restaurant_lat")
    @classmethod
    def validate_latitude(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {v}")
        return v

    @field_validator("courier_lon", "restaurant_lon")
    @classmethod
    def validate_longitude(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {v}")
        return v


class InferenceRecord(BaseModel):
    """Schema for inference input record (subset of full record)."""

    model_config = ConfigDict(strict=True)

    courier_lat: float = Field(..., ge=-90, le=90, description="Courier latitude")
    courier_lon: float = Field(..., ge=-180, le=180, description="Courier longitude")
    restaurant_lat: float = Field(..., ge=-90, le=90, description="Restaurant latitude")
    restaurant_lon: float = Field(
        ..., ge=-180, le=180, description="Restaurant longitude"
    )
    courier_location_timestamp: datetime = Field(
        ..., description="Timestamp of courier location"
    )


class DataFrameValidator:
    """Validator for pandas DataFrames using Pydantic schemas."""

    @staticmethod
    def validate_training_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
        """
        Validate training data DataFrame.

        Args:
            df: Input DataFrame to validate

        Returns:
            Tuple of (valid_df, errors) where errors is a list of validation errors
        """
        errors = []
        valid_rows = []

        # Check required columns
        missing_cols = set(REQUIRED_INPUT_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for idx, row in df.iterrows():
            try:
                # Parse timestamps if they're strings
                record_dict = row.to_dict()
                if isinstance(record_dict.get("courier_location_timestamp"), str):
                    record_dict["courier_location_timestamp"] = pd.to_datetime(
                        record_dict["courier_location_timestamp"]
                    )
                if isinstance(record_dict.get("order_created_timestamp"), str):
                    record_dict["order_created_timestamp"] = pd.to_datetime(
                        record_dict["order_created_timestamp"]
                    )

                CourierLocationRecord(**record_dict)
                valid_rows.append(idx)
            except Exception as e:
                errors.append({"row": idx, "error": str(e)})

        if errors:
            logger.warning(f"Found {len(errors)} invalid rows out of {len(df)}")
            for err in errors[:5]:  # Log first 5 errors
                logger.warning(f"Row {err['row']}: {err['error']}")

        valid_df = df.loc[valid_rows].copy()
        logger.info(
            f"Validation complete: {len(valid_df)} valid rows, {len(errors)} invalid rows"
        )

        return valid_df, errors

    @staticmethod
    def validate_inference_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
        """
        Validate inference input DataFrame.

        Args:
            df: Input DataFrame to validate

        Returns:
            Tuple of (valid_df, errors) where errors is a list of validation errors
        """
        errors = []
        valid_rows = []

        # Check required columns
        missing_cols = set(REQUIRED_INFERENCE_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for inference: {missing_cols}")

        for idx, row in df.iterrows():
            try:
                record_dict = row.to_dict()
                if isinstance(record_dict.get("courier_location_timestamp"), str):
                    record_dict["courier_location_timestamp"] = pd.to_datetime(
                        record_dict["courier_location_timestamp"]
                    )

                InferenceRecord(**record_dict)
                valid_rows.append(idx)
            except Exception as e:
                errors.append({"row": idx, "error": str(e)})

        if errors:
            logger.warning(f"Found {len(errors)} invalid rows for inference")

        valid_df = df.loc[valid_rows].copy()
        return valid_df, errors


def validate_dataframe(
    df: pd.DataFrame, validation_type: str = "training"
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Convenience function to validate a DataFrame.

    Args:
        df: DataFrame to validate
        validation_type: Either "training" or "inference"

    Returns:
        Tuple of (valid_df, errors)
    """
    validator = DataFrameValidator()

    if validation_type == "training":
        return validator.validate_training_data(df)
    elif validation_type == "inference":
        return validator.validate_inference_data(df)
    else:
        raise ValueError(f"Unknown validation type: {validation_type}")
