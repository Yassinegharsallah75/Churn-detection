"""
Data Ingestion Module
Handles downloading, loading, and initial validation of the Telco Customer Churn dataset.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_data(config: Optional[dict] = None) -> str:
    """
    Download the Telco Customer Churn dataset from Kaggle.
    Note: Requires kaggle.json credentials or manual download.
    
    For this project, we'll create a sample dataset loader that works
    with the Kaggle-downloaded file.
    """
    if config is None:
        config = load_config()
    
    raw_path = Path(config["data"]["raw_path"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    if raw_path.exists():
        logger.info(f"Data already exists at {raw_path}")
        return str(raw_path)
    
    logger.info("Please download the dataset from:")
    logger.info("https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    logger.info(f"And place it at: {raw_path}")
    
    # Create a placeholder message
    logger.warning("Dataset not found. Please download manually from Kaggle.")
    
    return str(raw_path)


def load_data(config: Optional[dict] = None) -> pd.DataFrame:
    """Load the raw dataset."""
    if config is None:
        config = load_config()
    
    raw_path = config["data"]["raw_path"]
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}.\n"
            "Please download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
    
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded dataset with shape: {df.shape}")
    return df


def validate_data(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[bool, list]:
    """
    Validate the dataset structure and content.
    Returns: (is_valid, list_of_issues)
    """
    if config is None:
        config = load_config()
    
    issues = []
    
    # Check required columns
    expected_cols = (
        config["features"]["categorical"] + 
        config["features"]["numerical"] + 
        [config["data"]["target_column"]] +
        config["features"]["drop_columns"]
    )
    
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for empty dataframe
    if df.empty:
        issues.append("Dataset is empty")
    
    # Check target distribution
    target_col = config["data"]["target_column"]
    if target_col in df.columns:
        churn_rate = df[target_col].value_counts(normalize=True)
        logger.info(f"Target distribution:\n{churn_rate}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found:\n{missing[missing > 0]}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate a summary of the dataset."""
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    return summary


if __name__ == "__main__":
    # Test the ingestion
    config = load_config()
    df = load_data(config)
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    is_valid, issues = validate_data(df, config)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    summary = get_data_summary(df)
    print(f"\nMemory Usage: {summary['memory_usage_mb']:.2f} MB")