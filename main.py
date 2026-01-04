import logging

from src.core.data_ingestion import load_and_process_data
from src.core.feature_engineering import generate_features
from src.core.model_training import train_model

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add file handler for logs
file_handler = logging.FileHandler("logs/pipeline.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)


def main():
    logger.info("Starting geo-busyness pipeline...")
    df, restaurants = load_and_process_data()
    df = generate_features(df, restaurants)
    _model = train_model(df)  # Model is saved to S3 by train_model
    logger.info("Pipeline completed!")


if __name__ == "__main__":
    main()
