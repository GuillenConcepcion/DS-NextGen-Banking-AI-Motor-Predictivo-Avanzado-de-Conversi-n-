import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from feature_engine.encoding import OneHotEncoder as FeOneHotEncoder
import logging

logger = logging.getLogger(__name__)

class MonthMapper(BaseEstimator, TransformerMixin):
    """Maps month abbreviations to numerical values."""
    def __init__(self):
        self.month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if 'month' in X.columns:
            X['month'] = X['month'].str.lower().map(self.month_map)
        return X

class BinaryMapper(BaseEstimator, TransformerMixin):
    """Maps 'yes'/'no' to 1/0."""
    def __init__(self, columns: list[str]):
        self.columns = columns
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                # Handle cases where it might already be numeric or boolean
                X[col] = X[col].replace({'no': 0, 'yes': 1, False: 0, True: 1}).astype(int)
        return X

class DynamicBinner(BaseEstimator, TransformerMixin):
    """Bins continuous variables into k categories based on Sturges' rule."""
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.k_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Sturges rule: k = 1 + 3.3 * log10(n)
        self.k_ = int(1 + 3.3 * np.log10(len(X)))
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns and self.k_ is not None:
                X[col] = pd.cut(X[col], bins=self.k_).astype(str)
        return X

class ContactImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values in the 'contact' column."""
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if 'contact' in X.columns:
            X['contact'] = X['contact'].fillna('unknown')
        return X

def get_simpler_model_pipeline() -> Pipeline:
    """
    Returns the Scikit-Learn pipeline for data preprocessing for the simpler model.
    """
    binary_cols = ['default', 'housing', 'loan']
    
    # Define the pipeline steps
    pipeline = Pipeline([
        ('binary_map', BinaryMapper(columns=binary_cols)),
        ('month_map', MonthMapper()),
        ('impute_contact', ContactImputer()),
        ('ohe_contact', FeOneHotEncoder(variables=['contact'], drop_last=True)),
        ('bin_pdays', DynamicBinner(columns=['pdays']))
    ])
    
    return pipeline

def select_and_order_columns(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Selects the required columns and ensures correct ordering for the simpler model.
    """
    expected_cols = [
        'default', 'housing', 'loan', 'day', 
        'contact_cellular', 'contact_telephone', 
        'month', 'campaign', 'pdays'
    ]
    
    if is_training and 'y' in df.columns:
        expected_cols.append('y')
        df['y'] = df['y'].replace({'no': 0, 'yes': 1})
        
    # Reindex to ensure all columns exist, fill missing one-hot columns with 0
    df = df.reindex(columns=expected_cols, fill_value=0)
    return df
