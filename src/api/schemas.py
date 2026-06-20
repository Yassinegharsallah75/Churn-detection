"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    """Input schema for a single customer prediction."""
    customerID: Optional[str] = Field(default="unknown", description="Customer ID")
    gender: str = Field(..., description="Male or Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, le=72, description="Months with company")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(
        ...,
        description="Electronic check, Mailed check, Bank transfer, or Credit card"
    )
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")


class PredictionResponse(BaseModel):
    """Output schema for prediction results."""
    customer_id: str
    will_churn: bool
    churn_probability: float
    retain_probability: float
    risk_level: str
    recommendation: str


class BatchPredictionRequest(BaseModel):
    """Input schema for batch predictions."""
    customers: List[CustomerData]


class BatchPredictionResponse(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response."""
    model_name: str
    model_version: str
    features: List[str]
    training_date: Optional[str] = None