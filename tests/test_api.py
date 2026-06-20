import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import pandas as pd

from api.app import app, get_predictor

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "service" in response.json()

@patch("api.app.get_predictor")
def test_model_info_endpoint(mock_get_predictor):
    mock_predictor = MagicMock()
    mock_predictor.get_model_info.return_value = {
        "model_type": "RandomForest",
        "f1_score": 0.88
    }
    mock_get_predictor.return_value = mock_predictor
    
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_type"] == "RandomForest"
    assert response.json()["f1_score"] == 0.88

@patch("api.app.get_predictor")
def test_predict_single_endpoint(mock_get_predictor):
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [{
        "prediction": 1,
        "probability": 0.95,
        "risk_score": 95.0
    }]
    mock_predictor.get_model_info.return_value = {"model_type": "XGBoost"}
    mock_get_predictor.return_value = mock_predictor
    
    # Payload matching TransactionInput
    payload = {
        "Time": 1.0,
        "Amount": 100.0,
        **{f"V{i}": 0.1 for i in range(1, 29)}
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["prediction"] == 1
    assert res_json["probability"] == 0.95
    assert res_json["risk_score"] == 95.0
    assert res_json["model_type"] == "XGBoost"

@patch("api.app.get_predictor")
def test_predict_batch_endpoint(mock_get_predictor):
    mock_predictor = MagicMock()
    mock_predictor.predict.return_value = [
        {"prediction": 0, "probability": 0.05, "risk_score": 5.0},
        {"prediction": 1, "probability": 0.98, "risk_score": 98.0}
    ]
    mock_predictor.get_model_info.return_value = {"model_type": "RandomForest"}
    mock_get_predictor.return_value = mock_predictor
    
    # Payload matching List[TransactionInput]
    payload = [
        {
            "Time": 1.0,
            "Amount": 5.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        },
        {
            "Time": 2.0,
            "Amount": 1000.0,
            **{f"V{i}": 1.5 for i in range(1, 29)}
        }
    ]
    
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    res_json = response.json()
    assert len(res_json) == 2
    assert res_json[0]["prediction"] == 0
    assert res_json[1]["prediction"] == 1
