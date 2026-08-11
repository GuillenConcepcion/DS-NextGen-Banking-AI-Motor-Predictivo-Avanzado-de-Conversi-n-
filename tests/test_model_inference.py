import pytest
import mlflow
import pandas as pd
from crispdm.utils.config import settings

@pytest.fixture
def dummy_batch():
    return pd.DataFrame([{
        "default": "no", "housing": "no", "loan": "no", "contact": "cellular", 
        "month": "may", "day": 15, "campaign": 2, "pdays": -1
    }])

def test_model_inference_pipeline(dummy_batch):
    """Test that the model registered in MLflow can successfully run inference."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    try:
        model = mlflow.pyfunc.load_model(f"models:/{settings.model_name}@{settings.model_alias}")
    except Exception as e:
        pytest.skip(f"MLflow model not available to test inference. Is MLflow running? Error: {e}")
        
    predictions = model.predict(dummy_batch)
    
    # Check that predictions are returned in probability array format
    assert len(predictions) == 1
    assert len(predictions[0]) == 2
    assert sum(predictions[0]) == 1.0
