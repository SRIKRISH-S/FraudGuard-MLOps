import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from src.ingestion import DataIngestion

@pytest.fixture
def sample_data():
    """
    Generates a minimal mock transaction dataframe.
    """
    columns = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    data = []
    
    # Add 10 non-fraud and 2 fraud transactions
    for i in range(12):
        row = [float(i), float(i * 10), 1 if i >= 10 else 0]
        # Append mock V1-V28 PCA values
        row.extend([np.random.normal() for _ in range(28)])
        data.append(row)
        
    return pd.DataFrame(data, columns=columns)

def test_validate_data_success(sample_data):
    ingestor = DataIngestion(raw_data_path="dummy.csv")
    assert ingestor.validate_data(sample_data) is True

def test_validate_data_missing_column(sample_data):
    ingestor = DataIngestion(raw_data_path="dummy.csv")
    invalid_data = sample_data.drop(columns=["V1"])
    assert ingestor.validate_data(invalid_data) is False

def test_split_and_save_data(sample_data):
    with tempfile.TemporaryDirectory() as temp_dir:
        ingestor = DataIngestion(raw_data_path="dummy.csv")
        train_df, test_df = ingestor.split_and_save_data(
            sample_data, 
            processed_dir=temp_dir, 
            test_size=0.25, 
            random_state=42
        )
        
        # Verify splits
        assert len(train_df) == 9
        assert len(test_df) == 3
        
        # Verify file existence
        assert os.path.exists(os.path.join(temp_dir, "train.csv"))
        assert os.path.exists(os.path.join(temp_dir, "test.csv"))
