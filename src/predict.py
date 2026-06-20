import os
import logging
from typing import Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelPrediction")

class FraudPredictor:
    """
    Class used to make real-time or batch inferences using the trained pipeline.
    """
    def __init__(
        self, 
        model_path: str = "models/best_model.pkl", 
        preprocessor_path: str = "models/preprocessor.pkl"
    ):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = None
        self.metadata = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """
        Loads the preprocessor, model, and metadata.
        """
        logger.info(f"Loading pipeline components: preprocessor={self.preprocessor_path}, model={self.model_path}")
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model file not found at {self.model_path}")
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        
        metadata_path = os.path.join(os.path.dirname(self.model_path), "model_metadata.pkl")
        if os.path.exists(metadata_path):
            self.metadata = joblib.load(metadata_path)
            logger.info(f"Model metadata loaded: {self.metadata}")
        else:
            logger.warning("Model metadata not found. Using default schema.")

    def predict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Runs prediction on a pandas DataFrame of transactions.
        
        Returns a list of dicts with predictions, probabilities, and risk scores.
        """
        logger.info(f"Received inference request for {len(df)} transactions.")
        
        # In case we get single dict/row, convert or ensure DataFrame format
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame([df])

        # Preprocess features
        X, _ = self.preprocessor.transform(df)
        
        # Compute predictions and probabilities
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        
        results = []
        for i in range(len(df)):
            prob = float(probs[i])
            pred = int(preds[i])
            # Risk score is probability mapped from 0 to 100
            risk_score = round(prob * 100, 2)
            
            results.append({
                "prediction": pred,
                "probability": prob,
                "risk_score": risk_score
            })
            
        logger.info(f"Successfully processed predictions for {len(results)} rows.")
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns loaded model metadata information.
        """
        if self.metadata:
            return self.metadata
        return {
            "model_type": str(type(self.model).__name__),
            "status": "Loaded"
        }
