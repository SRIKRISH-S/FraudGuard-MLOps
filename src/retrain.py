import os
import logging
import pandas as pd
from src.monitoring import DriftMonitor
from src.ingestion import DataIngestion
from src.train import ModelTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/retraining.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelRetraining")

class AutoRetrainer:
    """
    Automates model retraining when data drift is detected.
    """
    def __init__(
        self,
        raw_train_path: str = "data/raw/creditcard.csv",
        processed_dir: str = "data/processed"
    ):
        self.raw_train_path = raw_train_path
        self.processed_dir = processed_dir
        self.monitor = DriftMonitor(
            reference_data_path=os.path.join(processed_dir, "train.csv")
        )

    def trigger_retrain(self, new_data: pd.DataFrame, force: bool = False) -> bool:
        """
        Validates if retraining is needed due to data drift or if 'force' is True.
        
        If retraining is triggered, it runs the pipeline and updates models.
        """
        logger.info("Starting retraining trigger evaluation...")
        
        drift_detected = False
        drift_details = {}
        
        if not force:
            try:
                drift_detected, drift_details = self.monitor.run_drift_check(
                    new_data, 
                    report_html_path="logs/retrain_drift_report.html"
                )
                logger.info(f"Drift Analysis results: Drift Detected={drift_detected}, Summary={drift_details}")
            except Exception as e:
                logger.error(f"Error checking drift: {e}. Retraining will proceed as fallback.")
                drift_detected = True  # Proceed as fallback
        else:
            logger.info("Retraining forced. Skipping drift detection check.")
            drift_detected = True

        if drift_detected or force:
            logger.info("Retraining trigger activated. Proceeding with retraining pipeline...")
            
            # Load original baseline data
            if not os.path.exists(self.raw_train_path):
                raise FileNotFoundError(f"Baseline raw train data not found at {self.raw_train_path}")
            
            baseline_df = pd.read_csv(self.raw_train_path)
            
            # Combine baseline data and new inference/feedback data
            logger.info(f"Combining original training data ({len(baseline_df)} rows) with new data ({len(new_data)} rows)...")
            combined_df = pd.concat([baseline_df, new_data], ignore_index=True)
            
            # Run Ingestion split & save
            ingestor = DataIngestion(raw_data_path=self.raw_train_path)
            if not ingestor.validate_data(combined_df):
                logger.error("Combined data validation failed. Aborting retraining.")
                return False
                
            ingestor.split_and_save_data(combined_df, processed_dir=self.processed_dir)
            
            # Initialize ModelTrainer and run training pipeline
            logger.info("Initializing ModelTrainer for retraining run...")
            trainer = ModelTrainer(processed_dir=self.processed_dir)
            trainer.run_pipeline()
            
            # Re-load the baseline in DriftMonitor
            self.monitor.load_reference_data()
            logger.info("Model retraining successfully completed and registered in MLflow.")
            return True
        else:
            logger.info("No data drift detected. Model retraining is not required.")
            return False

if __name__ == "__main__":
    # Test execution
    try:
        # Load a small batch to simulate incoming production data
        train_path = "data/processed/train.csv"
        if os.path.exists(train_path):
            test_batch = pd.read_csv(train_path).sample(100, random_state=42)
            retrainer = AutoRetrainer()
            retrained = retrainer.trigger_retrain(test_batch, force=True)
            logger.info(f"Retrain test trigger completed. Retrained={retrained}")
    except Exception as e:
        logger.exception("Auto retraining execution failed.")
