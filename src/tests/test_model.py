"""
Unit tests for model training and prediction.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.models.train import get_model, evaluate_model


class TestModelInitialization:
    """Tests for model initialization."""
    
    def test_get_random_forest(self):
        model = get_model("RandomForestClassifier", {"n_estimators": 100})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 100
    
    def test_get_unknown_model(self):
        with pytest.raises(ValueError):
            get_model("UnknownModel")


class TestModelEvaluation:
    """Tests for model evaluation."""
    
    def test_evaluate_model(self):
        # Create dummy data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metrics = evaluate_model(model, X, y)
        
        # When no prefix is passed, keys have no prefix
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_evaluate_model_with_prefix(self):
        # Create dummy data
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        metrics = evaluate_model(model, X, y, prefix="test_")
        
        # When prefix is passed, keys include the prefix
        assert "test_accuracy" in metrics
        assert "test_precision" in metrics
        assert "test_recall" in metrics
        assert "test_f1" in metrics
        assert "test_roc_auc" in metrics
        assert 0 <= metrics["test_accuracy"] <= 1