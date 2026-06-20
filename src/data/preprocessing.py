"""
Data Preprocessing Module
Handles data cleaning, feature engineering, and transformation pipelines.
"""

import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer for cleaning the Telco dataset."""
    
    def __init__(self, drop_columns: List[str] = None):
        self.drop_columns = drop_columns or ["customerID"]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Drop ID columns
        df = df.drop(columns=[col for col in self.drop_columns if col in df.columns], errors="ignore")
        
        # Convert TotalCharges to numeric (handles empty strings)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # Fill missing TotalCharges with MonthlyCharges * tenure
            mask = df["TotalCharges"].isna()
            df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
        
        # Clean categorical values: "No internet service" -> "No"
        internet_cols = [
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        for col in internet_cols:
            if col in df.columns:
                df[col] = df[col].replace("No internet service", "No")
        
        if "MultipleLines" in df.columns:
            df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")
        
        return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for feature engineering."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Create new features
        if all(col in df.columns for col in ["tenure", "MonthlyCharges"]):
            df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
        
        if "tenure" in df.columns:
            df["TenureGroup"] = pd.cut(
                df["tenure"],
                bins=[0, 12, 24, 48, 72],
                labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
            )
        
        # Count number of services
        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"
        ]
        available_cols = [col for col in service_cols if col in df.columns]
        if available_cols:
            df["NumServices"] = (df[available_cols] == "Yes").sum(axis=1)
        
        return df


def create_preprocessing_pipeline(config: Optional[dict] = None) -> Pipeline:
    """Create the full preprocessing pipeline."""
    if config is None:
        config = load_config()
    
    categorical_features = config["features"]["categorical"]
    numerical_features = config["features"]["numerical"]
    drop_columns = config["features"]["drop_columns"]
    
    # Note: After feature engineering, we'll have additional columns
    # The actual categorical/numerical split will be handled dynamically
    
    pipeline = Pipeline([
        ("cleaner", DataCleaner(drop_columns=drop_columns)),
        ("feature_engineer", FeatureEngineer()),
    ])
    
    return pipeline


def prepare_features(df: pd.DataFrame, config: Optional[dict] = None, fit: bool = True):
    """
    Prepare features for modeling.
    Returns: X, y, preprocessor (fitted if fit=True)
    """
    if config is None:
        config = load_config()
    
    target_col = config["data"]["target_column"]
    
    # Separate target
    y = df[target_col].map({"Yes": 1, "No": 0}) if target_col in df.columns else None
    X = df.drop(columns=[target_col], errors="ignore")
    
    # Apply cleaning and feature engineering
    cleaner = DataCleaner(config["features"]["drop_columns"])
    X_clean = cleaner.fit_transform(X)
    
    engineer = FeatureEngineer()
    X_engineered = engineer.fit_transform(X_clean)
    
    # Identify final categorical and numerical columns
    categorical_cols = X_engineered.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
    
    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Numerical features: {numerical_cols}")
    
    # Create column transformer
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_cols),
        ("cat", "passthrough", categorical_cols),  # We'll one-hot encode separately or use model that handles it
    ], remainder="drop")
    
    if fit:
        X_processed = preprocessor.fit_transform(X_engineered)
    else:
        X_processed = preprocessor.transform(X_engineered)
    
    # Convert to DataFrame for easier handling
    feature_names = numerical_cols + categorical_cols
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_engineered.index)
    
    # One-hot encode categorical features
    X_final = pd.get_dummies(X_processed_df, columns=categorical_cols, drop_first=True)
    
    return X_final, y, preprocessor


def save_preprocessor(preprocessor, path: str):
    """Save the fitted preprocessor."""
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor saved to {path}")


def load_preprocessor(path: str):
    """Load a saved preprocessor."""
    return joblib.load(path)


if __name__ == "__main__":
    from src.data.ingestion import load_data
    
    config = load_config()
    df = load_data(config)
    
    X, y, preprocessor = prepare_features(df, config)
    print(f"\nProcessed features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nFeature columns: {list(X.columns)}")