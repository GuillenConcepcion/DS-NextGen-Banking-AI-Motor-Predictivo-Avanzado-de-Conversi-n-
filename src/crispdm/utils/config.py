from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    # Data configuration
    data_dir: Path = Field(default=Path("data"))
    
    # MLflow configuration
    mlflow_tracking_uri: str = Field(default="http://127.0.0.1:5000/")
    experiment_name: str = Field(default="CRISP_DM")
    model_name: str = Field(default="Catboost_Simpler_Cols")
    model_alias: str = Field(default="champion")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
