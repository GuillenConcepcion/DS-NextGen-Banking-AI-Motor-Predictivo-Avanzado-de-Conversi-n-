import pandas as pd
import logging
from pathlib import Path
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable
from crispdm.utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def monitor():
    data_path = settings.data_dir / "raw" / "bank_marketing.csv"
    if not data_path.exists():
        logger.error("Dataset not found! Please run 'make data'")
        return

    logger.info("Loading reference and current data for drift analysis...")
    df = pd.read_csv(data_path)
    
    # We drop the columns we don't use in V2 to avoid noise in the drift report
    unused_cols = ['age', 'job', 'marital', 'education', 'balance', 'previous', 'poutcome', 'duration']
    df = df.drop(columns=[col for col in unused_cols if col in df.columns], errors='ignore')
    
    # Simulate Reference vs Current data (e.g. 70% Reference, 30% Current production data)
    ref_size = int(len(df) * 0.7)
    reference_df = df.iloc[:ref_size]
    current_df = df.iloc[ref_size:]
    
    logger.info("Configuring Evidently AI Column Mapping...")
    column_mapping = ColumnMapping(
        target='y',
        prediction=None,
        numerical_features=['day', 'campaign', 'pdays'],
        categorical_features=['default', 'housing', 'loan', 'contact', 'month']
    )
    
    logger.info("Generating Data Drift Report...")
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping)
    
    output_path = "drift_report.html"
    report.save_html(output_path)
    logger.info(f"Drift report successfully saved to {output_path}")

if __name__ == "__main__":
    monitor()
