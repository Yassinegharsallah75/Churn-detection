"""
Model Training Module
Handles model training, hyperparameter tuning, and MLflow experiment tracking.
"""

import os
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import mlflow
import mlflow.sklearn
import joblib
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(model_name: str, params: Optional[Dict[str, Any]] = None):
    """Initialize a model by name."""
    models = {
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model_class = models[model_name]
    if params:
        return model_class(**params)
    return model_class()


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    config: Optional[dict] = None,
    model_name: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Train a model with cross-validation and MLflow tracking.
    """
    if config is None:
        config = load_config()
    
    model_name = model_name or config["model"]["name"]
    model_params = model_params or config["model"]["params"]
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run() as run:
        logger.info(f"Training {model_name} with params: {model_params}")
        
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model_params)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("num_features", X_train.shape[1])
        
        # Initialize model
        model = get_model(model_name, model_params)
        
        # Cross-validation
        cv = StratifiedKFold(
            n_splits=config["training"]["cv_folds"],
            shuffle=True,
            random_state=config["project"]["random_state"]
        )
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=config["training"]["scoring"]
        )
        
        logger.info(f"CV {config['training']['scoring']} scores: {cv_scores}")
        logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        mlflow.log_metric("cv_roc_auc_mean", cv_scores.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_scores.std())
        
        # Fit on full training data
        model.fit(X_train, y_train)
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            val_metrics = evaluate_model(model, X_val, y_val, prefix="val_")
            for metric_name, metric_value in val_metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log feature importance
        if hasattr(model, "feature_importances_"):
            feature_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            mlflow.log_table(data=feature_importance, artifact_file="feature_importance.json")
        
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        
        return model, run_id


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix: str = ""
) -> Dict[str, float]:
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        f"{prefix}accuracy": accuracy_score(y_test, y_pred),
        f"{prefix}precision": precision_score(y_test, y_pred),
        f"{prefix}recall": recall_score(y_test, y_pred),
        f"{prefix}f1": f1_score(y_test, y_pred),
        f"{prefix}roc_auc": roc_auc_score(y_test, y_pred_proba),
    }
    
    logger.info(f"\n{prefix.upper()} Evaluation Metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    logger.info(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    return metrics


def save_model(model, path: str):
    """Save trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    """Load a trained model."""
    return joblib.load(path)


if __name__ == "__main__":
    from src.data.ingestion import load_data
    from src.data.preprocessing import prepare_features
    from sklearn.model_selection import train_test_split
    
    config = load_config()
    
    # Load and preprocess data
    df = load_data(config)
    X, y, preprocessor = prepare_features(df, config)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["project"]["random_state"],
        stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config["data"]["validation_size"],
        random_state=config["project"]["random_state"],
        stratify=y_train
    )
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Train model
    model, run_id = train_model(X_train, y_train, X_val, y_val, config)
    
    # Final evaluation on test set
    test_metrics = evaluate_model(model, X_test, y_test, prefix="test_")
    
    # Save model
    save_model(model, "models/churn_model.pkl")