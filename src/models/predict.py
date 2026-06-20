"""
Prediction Module
Handles model inference for single and batch predictions with preprocessing.
"""

import logging
from typing import Union, Dict, List
import pandas as pd
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Replicate cleaning from preprocessing for API predictions."""
    
    @staticmethod
    def transform(df):
        df = df.copy()
        
        # Drop ID columns
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
        
        # Convert TotalCharges to numeric
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            mask = df["TotalCharges"].isna()
            df.loc[mask, "TotalCharges"] = df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
        
        # Clean categorical values
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


class FeatureEngineer:
    """Replicate feature engineering from preprocessing for API predictions."""
    
    @staticmethod
    def transform(df):
        df = df.copy()
        
        # Create new features
        if all(col in df.columns for col in ["tenure", "MonthlyCharges", "TotalCharges"]):
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


class ChurnPredictor:
    """Wrapper for churn prediction with full preprocessing."""
    
    def __init__(self, model_path: str, preprocessor_path: str = None):
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path) if preprocessor_path else None
        logger.info(f"Loaded model from {model_path}")
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full preprocessing pipeline."""
        # Step 1: Clean data
        df = DataCleaner.transform(df)
        
        # Step 2: Engineer features
        df = FeatureEngineer.transform(df)
        
        # Step 3: One-hot encode categorical features (same as training)
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Align with model features (fill missing with 0)
        if hasattr(self.model, 'feature_names_in_'):
            expected_features = self.model.feature_names_in_
            for col in expected_features:
                if col not in df_processed.columns:
                    df_processed[col] = 0
            df_processed = df_processed[expected_features]
        
        return df_processed
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict:
        """
        Make predictions on input data.
        
        Args:
            data: Input data as DataFrame, dict, or list of dicts
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Apply preprocessing
        df_processed = self._preprocess(df)
        
        # Make predictions
        predictions = self.model.predict(df_processed)
        probabilities = self.model.predict_proba(df_processed)
        
        results = {
            "predictions": predictions.tolist(),
            "churn_probability": probabilities[:, 1].tolist(),
            "retain_probability": probabilities[:, 0].tolist(),
            "churn_risk": [
                "High" if p > 0.7 else "Medium" if p > 0.4 else "Low"
                for p in probabilities[:, 1]
            ]
        }
        
        return results
    
    def predict_single(self, customer_data: Dict) -> Dict:
        """Predict for a single customer with detailed output."""
        result = self.predict(customer_data)
        
        return {
            "customer_id": customer_data.get("customerID", "unknown"),
            "will_churn": bool(result["predictions"][0]),
            "churn_probability": round(result["churn_probability"][0], 4),
            "retain_probability": round(result["retain_probability"][0], 4),
            "risk_level": result["churn_risk"][0],
            "recommendation": self._get_recommendation(result["churn_probability"][0])
        }
    
    def _get_recommendation(self, probability: float) -> str:
        """Generate business recommendation based on churn probability."""
        if probability > 0.8:
            return "URGENT: Immediate retention intervention required. Offer significant discount or premium support."
        elif probability > 0.6:
            return "HIGH: Proactive outreach recommended. Consider loyalty program or contract upgrade offers."
        elif probability > 0.4:
            return "MEDIUM: Monitor closely. Send satisfaction survey and targeted promotions."
        else:
            return "LOW: Customer is stable. Maintain regular engagement and service quality."


if __name__ == "__main__":
    # Example usage
    sample_customer = {
        "customerID": "7590-VHVEG",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    predictor = ChurnPredictor("models/churn_model.pkl")
    result = predictor.predict_single(sample_customer)
    print(result)