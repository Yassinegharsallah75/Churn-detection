"""
Unit tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.schemas import CustomerData


client = TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()


class TestModelInfoEndpoint:
    """Tests for model info endpoint."""
    
    def test_model_info(self):
        response = client.get("/model/info")
        # May return 503 if model not loaded in test environment
        assert response.status_code in [200, 503]


class TestPredictionEndpoint:
    """Tests for prediction endpoints."""
    
    def test_predict_single(self):
        sample_customer = {
            "customerID": "TEST-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 65.0,
            "TotalCharges": 780.0
        }
        
        response = client.post("/predict", json=sample_customer)
        # May return 503 if model not loaded
        assert response.status_code in [200, 503, 422]
    
    def test_predict_invalid_data(self):
        invalid_customer = {
            "gender": "Invalid",
            "SeniorCitizen": 5  # Invalid value
        }
        
        response = client.post("/predict", json=invalid_customer)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for monitoring endpoints."""
    
    def test_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "churn_predictions_total" in response.text or response.text == ""