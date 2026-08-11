import pytest
import requests

def test_podman_mlflow_health():
    """Test if MLflow tracking server is running (checking default compose port 5000)."""
    try:
        response = requests.get("http://localhost:5000", timeout=2)
        assert response.status_code == 200, "MLflow is running but returned non-200 status code."
    except requests.exceptions.ConnectionError:
        pytest.skip("Podman Compose MLflow is not running locally. Skipping test.")

def test_podman_streamlit_health():
    """Test if Streamlit App is running (checking default compose port 8501)."""
    try:
        response = requests.get("http://localhost:8501", timeout=2)
        # Streamlit returns 200 on /healthz
        assert response.status_code == 200, "Streamlit is running but returned non-200 status code."
    except requests.exceptions.ConnectionError:
        pytest.skip("Podman Compose Streamlit is not running locally. Skipping test.")
