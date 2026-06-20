"""
MLOps Pipeline Orchestration
End-to-end pipeline: Data → Preprocessing → Training → Evaluation → Registration
"""

import os
import logging
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.ingestion import load_data, validate_data, load_config
from src.data.preprocessing import prepare_features, save_preprocessor
from src.models.train import train_model, evaluate_model, save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_training_pipeline(
    config_path: str = "config.yaml",
    save_artifacts: bool = True
) -> Tuple[Optional[object], Optional[str], dict]:
    """
    Run the complete training pipeline.
    
    Returns:
        model: Trained model
        run_id: MLflow run ID
        metrics: Evaluation metrics
    """
    logger.info("=" * 60)
    logger.info("Starting MLOps Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load configuration
    logger.info("Step 1: Loading configuration...")
    config = load_config(config_path)
    
    # Step 2: Data Ingestion
    logger.info("Step 2: Loading data...")
    df = load_data(config)
    
    # Step 3: Data Validation
    logger.info("Step 3: Validating data...")
    is_valid, issues = validate_data(df, config)
    if not is_valid:
        logger.error(f"Data validation failed: {issues}")
        raise ValueError(f"Data validation failed: {issues}")
    logger.info("Data validation passed!")
    
    # Step 4: Preprocessing
    logger.info("Step 4: Preprocessing features...")
    X, y, preprocessor = prepare_features(df, config, fit=True)
    
    # Step 5: Train-Test Split
    logger.info("Step 5: Splitting data...")
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
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Step 6: Model Training
    logger.info("Step 6: Training model...")
    model, run_id = train_model(X_train, y_train, X_val, y_val, config)
    
    # Step 7: Final Evaluation
    logger.info("Step 7: Evaluating on test set...")
    test_metrics = evaluate_model(model, X_test, y_test, prefix="test_")
    
    # Step 8: Save Artifacts
    if save_artifacts:
        logger.info("Step 8: Saving artifacts...")
        os.makedirs("models", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
        save_model(model, "models/churn_model.pkl")
        save_preprocessor(preprocessor, "models/preprocessor.pkl")
        
        # Save reference data for drift detection
        X_train.to_csv("data/reference/reference_data.csv", index=False)
        logger.info("Artifacts saved!")
    
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info(f"MLflow Run ID: {run_id}")
    logger.info("=" * 60)
    
    return model, run_id, test_metrics


def run_batch_prediction(
    input_path: str,
    output_path: str,
    model_path: str = "models/churn_model.pkl",
    preprocessor_path: str = "models/preprocessor.pkl"
) -> pd.DataFrame:
    """
    Run batch prediction on new data.
    
    Args:
        input_path: Path to CSV file with customer data
        output_path: Path to save predictions
        model_path: Path to trained model
        preprocessor_path: Path to preprocessor
    """
    from src.models.predict import ChurnPredictor
    
    logger.info(f"Running batch prediction on {input_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    
    # Load predictor
    predictor = ChurnPredictor(model_path, preprocessor_path)
    
    # Predict
    results = predictor.predict(df)
    
    # Add predictions to dataframe
    df["churn_prediction"] = results["predictions"]
    df["churn_probability"] = results["churn_probability"]
    df["risk_level"] = results["churn_risk"]
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Run full pipeline
    model, run_id, metrics = run_training_pipeline()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"MLflow Run ID: {run_id}")
    print(f"Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"Test ROC-AUC: {metrics.get('test_roc_auc', 'N/A'):.4f}")