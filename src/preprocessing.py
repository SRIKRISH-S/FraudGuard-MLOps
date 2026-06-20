import os
import logging
from typing import Tuple, Union, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DataPreprocessing")

class DataPreprocessor:
    """
    Handles scaling of numerical columns, feature preparation, 
    and handles class imbalance for model training.
    """
    def __init__(self, apply_smote: bool = True, random_state: int = 42):
        self.apply_smote = apply_smote
        self.random_state = random_state
        self.amount_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self.features_v = [f"V{i}" for i in range(1, 29)]
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """
        Fits the scalers for 'Amount' and 'Time' columns.
        """
        logger.info("Fitting preprocessor (scalers) on input data...")
        if "Amount" not in df.columns or "Time" not in df.columns:
            raise KeyError("Input DataFrame must contain 'Amount' and 'Time' columns.")
        
        # Fit scalers
        self.amount_scaler.fit(df[["Amount"]])
        self.time_scaler.fit(df[["Time"]])
        
        self.fitted = True
        logger.info("Preprocessor successfully fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Applies scaling, combines V1-V28 features, and splits features (X) and label (y).
        """
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        
        logger.info("Transforming DataFrame...")
        
        # Check columns
        if "Amount" not in df.columns or "Time" not in df.columns:
            raise KeyError("DataFrame must contain 'Amount' and 'Time' columns for transformation.")
        for col in self.features_v:
            if col not in df.columns:
                raise KeyError(f"DataFrame is missing feature column: {col}")
        
        # Scale features
        scaled_amount = self.amount_scaler.transform(df[["Amount"]])
        scaled_time = self.time_scaler.transform(df[["Time"]])
        
        # Extract existing PCA components (V1-V28)
        v_features = df[self.features_v].values
        
        # Concatenate features: Scaled Time, PCA Features, Scaled Amount
        X = np.hstack([scaled_time, v_features, scaled_amount])
        
        y = None
        if "Class" in df.columns:
            y = df["Class"].values
            logger.info(f"Transformation complete. Returning X of shape {X.shape} and y of shape {y.shape}")
        else:
            logger.info(f"Transformation complete (no Class column). Returning X of shape {X.shape}")
            
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fits and transforms the input DataFrame in one call.
        """
        return self.fit(df).transform(df)

    def balance_training_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies SMOTE to training features and labels to handle class imbalance.
        """
        if not self.apply_smote:
            logger.info("SMOTE is disabled. Skipping class balancing.")
            return X, y
            
        logger.info(f"Applying SMOTE to balance data. Original counts: {np.bincount(y)}")
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"SMOTE applied. Resampled counts: {np.bincount(y_resampled)}")
            return X_resampled, y_resampled
        except Exception as e:
            logger.error(f"Error applying SMOTE: {e}. Returning original data.")
            return X, y

    def save(self, path: str = "models/preprocessor.pkl") -> None:
        """
        Saves the fitted preprocessor instance to a file.
        """
        logger.info(f"Saving preprocessor to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str = "models/preprocessor.pkl") -> "DataPreprocessor":
        """
        Loads a preprocessor instance from a file.
        """
        if not os.path.exists(path):
            error_msg = f"Preprocessor model file not found at {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        logger.info(f"Loading preprocessor from {path}")
        preprocessor = joblib.load(path)
        return preprocessor

if __name__ == "__main__":
    # Test script locally
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        preprocessor = DataPreprocessor(apply_smote=True)
        X_train, y_train = preprocessor.fit_transform(train_df)
        X_res, y_res = preprocessor.balance_training_data(X_train, y_train)
        preprocessor.save()
        logger.info("Preprocessor local test and saving passed.")
    except Exception as e:
        logger.exception("Preprocessing run failed.")
