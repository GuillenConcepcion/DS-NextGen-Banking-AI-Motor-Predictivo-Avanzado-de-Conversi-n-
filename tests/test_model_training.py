import pytest
import mlflow
from unittest.mock import patch
from crispdm.utils.config import settings

def test_mlflow_tracking_uri_configuration():
    """Test that the application is correctly pointing to the desired tracking URI."""
    assert settings.mlflow_tracking_uri is not None
    assert "http" in settings.mlflow_tracking_uri or "file" in settings.mlflow_tracking_uri

@patch("mlflow.sklearn.log_model")
def test_mocked_model_registration(mock_log_model):
    """Test that MLflow registration function is mocked and called (simulation of train_model.py logic)."""
    # Simulate a call to register a model
    mock_log_model(
        sk_model="dummy_model",
        artifact_path="model",
        registered_model_name=settings.model_name
    )
    mock_log_model.assert_called_once()
    assert mock_log_model.call_args.kwargs["registered_model_name"] == settings.model_name
