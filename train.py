#!/usr/bin/env python
"""
CLI Entry Point for Training
Usage: python train.py [--config config.yaml]
"""

import argparse
import logging
from src.pipeline.pipeline import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save artifacts"
    )
    
    args = parser.parse_args()
    
    logger.info("Starting training pipeline...")
    model, run_id, metrics = run_training_pipeline(
        config_path=args.config,
        save_artifacts=not args.no_save
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"MLflow Run ID: {run_id}")
    print(f"Test Accuracy: {metrics.get('test_accuracy', 'N/A'):.4f}")
    print(f"Test ROC-AUC: {metrics.get('test_roc_auc', 'N/A'):.4f}")
    print(f"Test F1-Score: {metrics.get('test_f1', 'N/A'):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()