"""
FastAPI Application for Churn Prediction
Production-ready API with monitoring, health checks, and batch processing.
"""

import os
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import (
    CustomerData, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse, ModelInfoResponse
)
from src.models.predict import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "churn_predictions_total",
    "Total number of predictions",
    ["risk_level"]
)
PREDICTION_LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "Prediction latency in seconds"
)
API_REQUESTS = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

# Global predictor instance
predictor: Optional[ChurnPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global predictor
    
    # Startup
    logger.info("Loading model...")
    model_path = os.getenv("MODEL_PATH", "models/churn_model.pkl")
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
    
    try:
        predictor = ChurnPredictor(model_path, preprocessor_path)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="MLOps-powered customer churn prediction service",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware to track request metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    API_REQUESTS.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Churn Prediction API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        model_name="RandomForestClassifier",
        model_version="1.0.0",
        features=[
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges"
        ]
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerData):
    """
    Predict churn for a single customer.
    
    Returns churn probability, risk level, and business recommendation.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with PREDICTION_LATENCY.time():
        try:
            result = predictor.predict_single(customer.dict())
            
            # Update metrics
            PREDICTION_COUNTER.labels(risk_level=result["risk_level"]).inc()
            
            return PredictionResponse(**result)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers.
    
    Efficient batch processing for bulk predictions.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        customers_data = [c.dict() for c in request.customers]
        results = predictor.predict(customers_data)
        
        predictions = []
        for i, customer in enumerate(request.customers):
            pred = {
                "customer_id": customer.customerID,
                "will_churn": bool(results["predictions"][i]),
                "churn_probability": round(results["churn_probability"][i], 4),
                "retain_probability": round(results["retain_probability"][i], 4),
                "risk_level": results["churn_risk"][i],
                "recommendation": predictor._get_recommendation(results["churn_probability"][i])
            }
            predictions.append(PredictionResponse(**pred))
            PREDICTION_COUNTER.labels(risk_level=pred["risk_level"]).inc()
        
        risk_counts = {"High": 0, "Medium": 0, "Low": 0}
        for p in predictions:
            risk_counts[p.risk_level] = risk_counts.get(p.risk_level, 0) + 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=risk_counts["High"],
            medium_risk_count=risk_counts["Medium"],
            low_risk_count=risk_counts["Low"]
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)