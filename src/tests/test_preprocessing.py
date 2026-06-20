"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import DataCleaner, FeatureEngineer, create_preprocessing_pipeline


class TestDataCleaner:
    """Tests for DataCleaner transformer."""
    
    def test_drop_columns(self):
        df = pd.DataFrame({
            "customerID": ["1", "2"],
            "gender": ["Male", "Female"],
            "tenure": [1, 2]
        })
        cleaner = DataCleaner(drop_columns=["customerID"])
        result = cleaner.fit_transform(df)
        assert "customerID" not in result.columns
        assert "gender" in result.columns
    
    def test_total_charges_conversion(self):
        df = pd.DataFrame({
            "TotalCharges": ["29.85", " "],
            "MonthlyCharges": [29.85, 50.0],
            "tenure": [1, 2]
        })
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert result["TotalCharges"].dtype == np.float64
        assert not result["TotalCharges"].isna().any()
    
    def test_internet_service_cleanup(self):
        df = pd.DataFrame({
            "OnlineSecurity": ["Yes", "No internet service"],
            "OnlineBackup": ["No", "No internet service"]
        })
        cleaner = DataCleaner()
        result = cleaner.fit_transform(df)
        assert "No internet service" not in result["OnlineSecurity"].values
        assert result["OnlineSecurity"].iloc[1] == "No"


class TestFeatureEngineer:
    """Tests for FeatureEngineer transformer."""
    
    def test_avg_monthly_charge(self):
        df = pd.DataFrame({
            "tenure": [12, 24],
            "MonthlyCharges": [50.0, 75.0],
            "TotalCharges": [600.0, 1800.0]
        })
        engineer = FeatureEngineer()
        result = engineer.fit_transform(df)
        assert "AvgMonthlyCharge" in result.columns
        assert result["AvgMonthlyCharge"].iloc[0] == 600.0 / 13
    
    def test_tenure_group(self):
        df = pd.DataFrame({"tenure": [6, 18, 36, 60]})
        engineer = FeatureEngineer()
        result = engineer.fit_transform(df)
        assert "TenureGroup" in result.columns
        assert len(result["TenureGroup"].cat.categories) == 4


class TestPreprocessingPipeline:
    """Tests for the full preprocessing pipeline."""
    
    def test_pipeline_creation(self):
        pipeline = create_preprocessing_pipeline()
        assert pipeline is not None
        assert "cleaner" in [name for name, _ in pipeline.steps]
        assert "feature_engineer" in [name for name, _ in pipeline.steps]