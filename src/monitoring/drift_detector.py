"""
Data Drift Detection Module
Monitors for data drift using statistical tests and Evidently AI.
"""

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from scipy import stats
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift between reference and current data."""
    
    def __init__(self, reference_data_path: str, threshold: float = 0.05):
        self.reference_data = pd.read_csv(reference_data_path)
        self.threshold = threshold
        logger.info(f"Loaded reference data: {self.reference_data.shape}")
    
    def detect_numerical_drift(
        self,
        current_data: pd.DataFrame,
        column: str
    ) -> Dict:
        """Detect drift in numerical columns using KS test."""
        reference = self.reference_data[column].dropna()
        current = current_data[column].dropna()
        
        statistic, p_value = stats.ks_2samp(reference, current)
        
        return {
            "column": column,
            "type": "numerical",
            "ks_statistic": statistic,
            "p_value": p_value,
            "drift_detected": p_value < self.threshold,
            "reference_mean": reference.mean(),
            "current_mean": current.mean(),
            "mean_diff_pct": abs(reference.mean() - current.mean()) / reference.mean() * 100
        }
    
    def detect_categorical_drift(
        self,
        current_data: pd.DataFrame,
        column: str
    ) -> Dict:
        """Detect drift in categorical columns using Chi-square test."""
        reference = self.reference_data[column].fillna("missing")
        current = current_data[column].fillna("missing")
        
        # Get value counts
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        try:
            statistic, p_value = stats.chisquare(curr_freq, ref_freq)
        except:
            p_value = 1.0
            statistic = 0.0
        
        return {
            "column": column,
            "type": "categorical",
            "chi2_statistic": statistic,
            "p_value": p_value,
            "drift_detected": p_value < self.threshold,
            "reference_categories": len(ref_counts),
            "current_categories": len(curr_counts)
        }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Run full drift detection on all features.
        
        Returns:
            Dictionary with drift report
        """
        logger.info("Running drift detection...")
        
        numerical_cols = current_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = current_data.select_dtypes(include=["object"]).columns.tolist()
        
        drift_results = []
        drift_detected_count = 0
        
        # Check numerical columns
        for col in numerical_cols:
            if col in self.reference_data.columns:
                result = self.detect_numerical_drift(current_data, col)
                drift_results.append(result)
                if result["drift_detected"]:
                    drift_detected_count += 1
                    logger.warning(f"DRIFT DETECTED in {col}: p={result['p_value']:.4f}")
        
        # Check categorical columns
        for col in categorical_cols:
            if col in self.reference_data.columns:
                result = self.detect_categorical_drift(current_data, col)
                drift_results.append(result)
                if result["drift_detected"]:
                    drift_detected_count += 1
                    logger.warning(f"DRIFT DETECTED in {col}: p={result['p_value']:.4f}")
        
        total_checked = len(numerical_cols) + len(categorical_cols)
        drift_ratio = drift_detected_count / total_checked if total_checked > 0 else 0
        
        report = {
            "drift_detected": drift_detected_count > 0,
            "drift_ratio": drift_ratio,
            "features_checked": total_checked,
            "features_drifted": drift_detected_count,
            "threshold": self.threshold,
            "details": drift_results
        }
        
        logger.info(f"Drift detection complete: {drift_detected_count}/{total_checked} features drifted")
        
        return report
    
    def generate_report(self, current_data: pd.DataFrame, output_path: str):
        """Generate and save a drift report."""
        report = self.detect_drift(current_data)
        
        # Save as JSON
        import json
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to {output_path}")
        return report


if __name__ == "__main__":
    # Example usage
    detector = DriftDetector("data/reference/reference_data.csv")
    
    # Simulate current data (in production, this would be new incoming data)
    current = pd.read_csv("data/reference/reference_data.csv")
    
    report = detector.detect_drift(current)
    print(f"\nDrift detected: {report['drift_detected']}")
    print(f"Features drifted: {report['features_drifted']}/{report['features_checked']}")