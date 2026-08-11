import pandas as pd
from crispdm.utils.config import settings

def test_raw_data_exists():
    """Verify that the raw dataset is available (e.g. downloaded by make data or pulled via DVC)"""
    data_path = settings.data_dir / "raw" / "bank_marketing.csv"
    assert data_path.exists(), f"Raw dataset missing at {data_path}!"

def test_data_schema():
    """Validate that the dataset conforms to the schema expected by the V2 model"""
    data_path = settings.data_dir / "raw" / "bank_marketing.csv"
    df = pd.read_csv(data_path)
    
    # Check essential V2 features are present
    expected_cols = ['day', 'campaign', 'pdays', 'default', 'housing', 'loan', 'contact', 'month', 'y']
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: {col}"
        
    # Check no missing values in these critical columns (contact may have nulls in raw data)
    cols_to_check = [c for c in expected_cols if c != 'contact']
    assert not df[cols_to_check].isnull().any().any(), "Null values found in critical columns!"
