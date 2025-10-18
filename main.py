"""
Main pipeline script for IRIS classification.
This script orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import os
import argparse
import logging
from pathlib import Path

from src.data_processing import DataProcessor
from src.model_training import ModelTrainer
from src.dvc_operations import DVCOperations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='IRIS Classification Pipeline')
    parser.add_argument('--data-path', type=str, default='iris-dvc-pipeline/v1_data.csv',
                       help='Path to the data file')
    parser.add_argument('--model-path', type=str, default='iris-dvc-pipeline/model.joblib',
                       help='Path to save the model')
    parser.add_argument('--metrics-path', type=str, default='iris-dvc-pipeline/metrics.txt',
                       help='Path to save the metrics')
    parser.add_argument('--augment-data', action='store_true',
                       help='Whether to augment the data')
    parser.add_argument('--version', type=str, default='v1.0',
                       help='DVC version to checkout')
    parser.add_argument('--setup-dvc', action='store_true',
                       help='Setup DVC remote and pull data/model from GCS')
    
    args = parser.parse_args()
    
    try:
        # Initialize components
        data_processor = DataProcessor()
        model_trainer = ModelTrainer()
        dvc_ops = DVCOperations()
        
        # Setup DVC and pull from GCS if requested
        if args.setup_dvc:
            logger.info("Setting up DVC remote and pulling data/model from GCS...")
            # Setup DVC remote
            remote_url = "gs://mlops-course-verdant-victory-473118-k0-unique-week2-2/iris-pipeline"
            if not dvc_ops.setup_remote(remote_url):
                logger.error("Failed to setup DVC remote")
                return 1
            
            # Pull data and model
            if not dvc_ops.pull_data(args.data_path):
                logger.error("Failed to pull data from DVC")
                return 1
            if not dvc_ops.pull_model(args.model_path):
                logger.error("Failed to pull model from DVC")
                return 1
        
        # Checkout specific version if requested
        if args.version != 'v1.0':
            logger.info(f"Checking out version {args.version}...")
            if not dvc_ops.checkout_version(args.version):
                logger.error(f"Failed to checkout version {args.version}")
                return 1
        
        # Load and validate data
        logger.info("Loading and validating data...")
        data = data_processor.load_data(args.data_path)
        
        if not data_processor.validate_data(data):
            logger.error("Data validation failed")
            return 1
        
        # Augment data if requested
        if args.augment_data:
            logger.info("Augmenting data...")
            data = data_processor.augment_data(data)
        
        # Split data
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = data_processor.split_data(data)
        
        # Train model
        logger.info("Training model...")
        model = model_trainer.train_model(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics = model_trainer.evaluate_model(X_test, y_test)
        
        # Save model and metrics
        logger.info("Saving model and metrics...")
        model_trainer.save_model(args.model_path)
        model_trainer.save_metrics(metrics, args.metrics_path)
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
