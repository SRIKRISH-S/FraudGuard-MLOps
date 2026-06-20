import os
import logging
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/ingestion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataIngestion")

class DataIngestion:
    """
    Handles data loading, validation, and splitting for the ML pipeline.
    """
    def __init__(self, raw_data_path: str = "data/raw/creditcard.csv"):
        self.raw_data_path = raw_data_path
        self.expected_columns = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]

    def load_raw_data(self) -> pd.DataFrame:
        """
        Loads raw credit card dataset from CSV path.
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        if not os.path.exists(self.raw_data_path):
            error_msg = f"Raw data file not found at {self.raw_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Successfully loaded dataset of shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading raw data: {e}")
            raise e

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validates the schema, checks for missing values, and checks class balance.
        """
        logger.info("Starting data validation check...")
        
        # Schema validation
        missing_cols = [col for col in self.expected_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Schema Validation Failed. Missing columns: {missing_cols}")
            return False
        
        # Null value check
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        if total_nulls > 0:
            logger.warning(f"Data contains null values: \n{null_counts[null_counts > 0]}")
            # If nulls are too high, raise warning, but we can drop or fill
            # Here we expect the Credit Card dataset to have 0 null values
        else:
            logger.info("Data validation passed: No missing values found.")

        # Check target class column
        if "Class" not in df.columns:
            logger.error("Target column 'Class' is missing.")
            return False
            
        class_counts = df["Class"].value_counts()
        if len(class_counts) < 2:
            logger.error("Dataset contains only one class. Training requires both fraud and non-fraud classes.")
            return False
            
        fraud_ratio = class_counts.get(1, 0) / len(df)
        logger.info(f"Class distribution: Non-Fraud = {class_counts.get(0, 0)}, Fraud = {class_counts.get(1, 0)} ({fraud_ratio:.4%})")
        
        logger.info("Data validation completed successfully.")
        return True

    def split_and_save_data(
        self, 
        df: pd.DataFrame, 
        processed_dir: str = "data/processed", 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data into train and test sets using stratification, then saves them to processed folder.
        """
        logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
        
        # Stratified train/test split to preserve fraud ratio
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["Class"]
        )
        
        logger.info(f"Split results - Train set: {train_df.shape}, Test set: {test_df.shape}")
        
        # Ensure processed directory exists
        os.makedirs(processed_dir, exist_ok=True)
        
        train_path = os.path.join(processed_dir, "train.csv")
        test_path = os.path.join(processed_dir, "test.csv")
        
        try:
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            logger.info(f"Saved splits to {train_path} and {test_path}")
        except Exception as e:
            logger.error(f"Error saving split files: {e}")
            raise e
            
        return train_df, test_df

if __name__ == "__main__":
    # Test script locally
    ingestor = DataIngestion()
    try:
        raw_df = ingestor.load_raw_data()
        if ingestor.validate_data(raw_df):
            ingestor.split_and_save_data(raw_df)
    except Exception as e:
        logger.exception("Ingestion run failed.")
