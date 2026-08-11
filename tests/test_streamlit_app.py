from streamlit.testing.v1 import AppTest
from unittest.mock import patch

def test_streamlit_app_loads_successfully():
    """Test the Streamlit app E2E rendering using the official AppTest framework."""
    at = AppTest.from_file("src/crispdm/app/finapp.py")
    
    with patch("crispdm.app.finapp.load_model_from_mlflow") as mock_load:
        at.run(timeout=15)
        
    # Check if there are no exceptions generated during rendering
    assert not at.exception, f"Streamlit encountered an unhandled exception during rendering: {at.exception}"
    
    # Check if the title exists
    assert len(at.title) > 0, "Title is missing from the Streamlit app."
    assert "Banking Conversion AI" in at.title[0].value
