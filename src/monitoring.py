import os
import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DriftMonitoring")

class DriftMonitor:
    """
    Handles data drift checking using Evidently AI reports.
    """
    def __init__(self, reference_data_path: str = "data/processed/train.csv"):
        self.reference_data_path = reference_data_path
        self.reference_df = None
        self.load_reference_data()

    def load_reference_data(self) -> None:
        """
        Loads the reference dataset used for baseline comparisons.
        """
        logger.info(f"Loading reference data for drift monitoring from {self.reference_data_path}")
        if not os.path.exists(self.reference_data_path):
            logger.warning(f"Reference train data not found at {self.reference_data_path}. Baseline comparisons will fail until model is trained.")
            return
        
        self.reference_df = pd.read_csv(self.reference_data_path)
        logger.info(f"Loaded reference dataset of shape: {self.reference_df.shape}")

    def run_drift_check(
        self, 
        current_df: pd.DataFrame, 
        report_html_path: str = "logs/drift_report.html"
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Compares reference dataset with a current dataset to detect feature drift.
        
        Returns:
            drift_detected: bool, True if percentage of drifted features exceeds threshold.
            drift_summary: dict, details about the drift test run.
        """
        if self.reference_df is None:
            self.load_reference_data()
            if self.reference_df is None:
                raise FileNotFoundError("Reference data is required for data drift check.")
                
        logger.info("Initializing Evidently AI Data Drift Report...")
        
        # Align columns to prevent metric mismatches
        common_cols = [col for col in self.reference_df.columns if col in current_df.columns]
        if not common_cols:
            raise KeyError("No common columns found between reference and current datasets.")
            
        ref_subset = self.reference_df[common_cols]
        cur_subset = current_df[common_cols]
        
        # Instantiate Evidently Report with DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        logger.info("Running drift analysis...")
        report.run(reference_data=ref_subset, current_data=cur_subset)
        
        # Save HTML dashboard
        os.makedirs(os.path.dirname(report_html_path), exist_ok=True)
        report.save_html(report_html_path)
        logger.info(f"Evidently HTML report saved to {report_html_path}")
        
        # Parse results
        report_dict = report.as_dict()
        
        # Extract metrics
        drift_metrics = report_dict["metrics"][0]["result"]
        number_of_features = drift_metrics["number_of_columns"]
        drifted_features = drift_metrics["number_of_drifted_columns"]
        share_of_drifted_features = drift_metrics["share_of_drifted_columns"]
        dataset_drift = drift_metrics["dataset_drift"]
        
        drift_summary = {
            "number_of_features": number_of_features,
            "drifted_features": drifted_features,
            "share_of_drifted_features": float(share_of_drifted_features),
            "dataset_drift_detected": bool(dataset_drift)
        }
        
        logger.info(f"Drift check summary: {drift_summary}")
        return bool(dataset_drift), drift_summary

if __name__ == "__main__":
    # Local test
    try:
        train = pd.read_csv("data/processed/train.csv")
        # Simulate drifted data by adding noise
        drifted_train = train.copy()
        drifted_train["Amount"] = drifted_train["Amount"] * 3.5 + 50.0
        
        monitor = DriftMonitor()
        drift_detected, summary = monitor.run_drift_check(drifted_train)
        logger.info(f"Test Run: Drift Detected={drift_detected}, Summary={summary}")
    except Exception as e:
        logger.exception("Monitoring run failed.")
