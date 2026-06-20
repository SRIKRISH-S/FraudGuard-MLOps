import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor

@pytest.fixture
def dummy_train_df():
    columns = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    # Create 100 rows, 10 fraud and 90 non-fraud
    data = []
    for i in range(100):
        row = [float(i), float(i * 5), 1 if i < 10 else 0]
        row.extend([np.random.normal() for _ in range(28)])
        data.append(row)
    return pd.DataFrame(data, columns=columns)

def test_fit_transform_shape(dummy_train_df):
    preprocessor = DataPreprocessor(apply_smote=False)
    X, y = preprocessor.fit_transform(dummy_train_df)
    
    # 30 features = 1 scaled Time + 28 PCA features + 1 scaled Amount
    assert X.shape == (100, 30)
    assert len(y) == 100

def test_balance_training_data(dummy_train_df):
    preprocessor = DataPreprocessor(apply_smote=True)
    X, y = preprocessor.fit_transform(dummy_train_df)
    
    # Verify imbalance before SMOTE
    assert np.sum(y == 1) == 10
    assert np.sum(y == 0) == 90
    
    # Apply SMOTE
    X_res, y_res = preprocessor.balance_training_data(X, y)
    
    # Verify class counts are equal after resampling
    assert np.sum(y_res == 1) == 90
    assert np.sum(y_res == 0) == 90

def test_preprocessor_save_load(dummy_train_df):
    preprocessor = DataPreprocessor()
    preprocessor.fit(dummy_train_df)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "preprocessor.pkl")
        preprocessor.save(save_path)
        
        # Load back
        loaded = DataPreprocessor.load(save_path)
        assert loaded.fitted is True
        assert loaded.amount_scaler is not None
        assert loaded.time_scaler is not None
