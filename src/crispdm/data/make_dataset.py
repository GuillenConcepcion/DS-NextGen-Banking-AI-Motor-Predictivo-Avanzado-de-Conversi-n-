import logging
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

from crispdm.utils.config import settings

logger = logging.getLogger(__name__)

def fetch_and_save_data(output_path: Path) -> None:
    """
    Fetches the Bank Marketing dataset from UCIML Repo and saves it to the raw data directory.
    """
    logger.info("Fetching Bank Marketing dataset (ID: 222)...")
    try:
        # Fetch dataset
        bank_marketing = fetch_ucirepo(id=222)
        
        # Combine features and targets
        df = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)
        
        # Rename specific columns to match existing convention
        df = df.rename(columns={'day_of_week': 'day'})
        
        # Ensure the directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save raw data
        logger.info(f"Saving raw data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info("Data fetching and saving completed successfully.")
        
    except Exception as e:
        logger.error(f"Error fetching or saving data: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw_data_path = settings.data_dir / "raw" / "bank_marketing.csv"
    fetch_and_save_data(raw_data_path)
