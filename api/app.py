import os
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd

from src.predict import FraudPredictor

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("API")

# Initialize FastAPI
app = FastAPI(
    title="FraudGuard API",
    description="Real-Time Credit Card Fraud Detection with MLOps",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Prediction Audit Log path
AUDIT_LOG_PATH = "logs/prediction_audit.csv"

# Global Predictor instance (lazy loaded)
predictor = None

def get_predictor() -> FraudPredictor:
    global predictor
    if predictor is None:
        try:
            predictor = FraudPredictor()
        except Exception as e:
            logger.error(f"Failed to load prediction artifacts: {e}")
            raise RuntimeError(f"Model artifacts not initialized. Please run training pipeline first. Error: {e}")
    return predictor

# Pydantic Schemas for validation
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction", example=0.0)
    Amount: float = Field(..., description="Transaction amount", example=149.62)
    # V1 to V28 PCA components
    V1: float = Field(..., example=-1.3598071336738)
    V2: float = Field(..., example=-0.07278117330985)
    V3: float = Field(..., example=2.53634673796914)
    V4: float = Field(..., example=1.37815522427443)
    V5: float = Field(..., example=-0.33832076994252)
    V6: float = Field(..., example=0.46238777776229)
    V7: float = Field(..., example=0.23959855406126)
    V8: float = Field(..., example=0.09869790126105)
    V9: float = Field(..., example=0.36378696961121)
    V10: float = Field(..., example=0.09079417197893)
    V11: float = Field(..., example=-0.55159953326081)
    V12: float = Field(..., example=-0.61780085576235)
    V13: float = Field(..., example=-0.99138984723541)
    V14: float = Field(..., example=-0.31116935369988)
    V15: float = Field(..., example=1.46817697209427)
    V16: float = Field(..., example=-0.47040052525948)
    V17: float = Field(..., example=0.20797124192924)
    V18: float = Field(..., example=0.02579058019856)
    V19: float = Field(..., example=0.40399296025573)
    V20: float = Field(..., example=0.25141209823971)
    V21: float = Field(..., example=-0.01830677794415)
    V22: float = Field(..., example=0.2778375755589)
    V23: float = Field(..., example=-0.11047391018877)
    V24: float = Field(..., example=0.06692807491467)
    V25: float = Field(..., example=0.12853935827353)
    V26: float = Field(..., example=-0.18911484388882)
    V27: float = Field(..., example=0.13355837674039)
    V28: float = Field(..., example=-0.02105305345382)

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for non-fraud, 1 for fraud")
    probability: float = Field(..., description="Fraud probability (0.0 to 1.0)")
    risk_score: float = Field(..., description="Fraud risk score (0 to 100)")
    model_type: str = Field(..., description="Model architecture used")

def log_prediction_to_csv(tx_data: Dict[str, Any], pred_data: Dict[str, Any]) -> None:
    """
    Appends predictions and transaction highlights to the prediction audit CSV.
    """
    audit_row = {
        "timestamp": datetime.now().isoformat(),
        "tx_time": tx_data.get("Time"),
        "tx_amount": tx_data.get("Amount"),
        "prediction": pred_data.get("prediction"),
        "probability": pred_data.get("probability"),
        "risk_score": pred_data.get("risk_score")
    }
    df_row = pd.DataFrame([audit_row])
    
    # Write header if file does not exist
    header = not os.path.exists(AUDIT_LOG_PATH)
    try:
        df_row.to_csv(AUDIT_LOG_PATH, mode="a", index=False, header=header)
    except Exception as e:
        logger.error(f"Failed to log transaction to audit file: {e}")

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Performs system health check.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "FraudGuard API"
    }

@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    """
    Returns metadata about the active ML model.
    """
    try:
        pred_service = get_predictor()
        return pred_service.get_model_info()
    except Exception as e:
        logger.error(f"Error loading model info: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def predict_single(transaction: TransactionInput) -> Dict[str, Any]:
    """
    Evaluates a single credit card transaction for fraud risk.
    """
    try:
        pred_service = get_predictor()
        input_data = transaction.dict()
        input_df = pd.DataFrame([input_data])
        
        # Run inference
        results = pred_service.predict(input_df)
        result = results[0]
        
        # Log prediction to audit CSV
        log_prediction_to_csv(input_data, result)
        
        # Include model type in response
        model_type = pred_service.get_model_info().get("model_type", "Unknown")
        
        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "risk_score": result["risk_score"],
            "model_type": model_type
        }
    except Exception as e:
        logger.exception("Error executing single prediction.")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(transactions: List[TransactionInput]) -> List[Dict[str, Any]]:
    """
    Evaluates a batch of credit card transactions for fraud risk.
    """
    try:
        pred_service = get_predictor()
        input_records = [tx.dict() for tx in transactions]
        input_df = pd.DataFrame(input_records)
        
        # Run inference
        results = pred_service.predict(input_df)
        
        # Log each prediction to audit CSV
        model_type = pred_service.get_model_info().get("model_type", "Unknown")
        for i, result in enumerate(results):
            log_prediction_to_csv(input_records[i], result)
            result["model_type"] = model_type
            
        return results
    except Exception as e:
        logger.exception("Error executing batch prediction.")
        raise HTTPException(status_code=500, detail=str(e))
