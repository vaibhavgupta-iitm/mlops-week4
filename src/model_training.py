"""
Model training module for IRIS pipeline.
Handles model training, evaluation, and saving.
"""

import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training and evaluation operations."""
    
    def __init__(self, max_depth: int = 3, random_state: int = 1):
        """
        Initialize ModelTrainer.
        
        Args:
            max_depth: Maximum depth of the decision tree
            random_state: Random seed for reproducibility
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
    
    def train_model(self, X_train, y_train) -> DecisionTreeClassifier:
        """
        Train the decision tree model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        try:
            self.model = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                random_state=self.random_state
            )
            self.model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
            return self.model
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test) -> Dict[str, Any]:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            metrics = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            logger.info(f"Model evaluation completed - Accuracy: {accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path where to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> DecisionTreeClassifier:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from: {model_path}")
            return self.model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def save_metrics(self, metrics: Dict[str, Any], metrics_path: str) -> None:
        """
        Save evaluation metrics to a text file.
        
        Args:
            metrics: Dictionary containing metrics
            metrics_path: Path where to save the metrics
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            
            with open(metrics_path, "w") as f:
                f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"Classification Report:\n")
                for class_name, class_metrics in metrics['classification_report'].items():
                    if isinstance(class_metrics, dict):
                        f.write(f"  {class_name}:\n")
                        for metric_name, value in class_metrics.items():
                            f.write(f"    {metric_name}: {value:.4f}\n")
                    else:
                        f.write(f"  {class_name}: {class_metrics:.4f}\n")
            
            logger.info(f"Metrics saved to: {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise
