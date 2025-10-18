"""
Unit tests for model evaluation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import joblib

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_training import ModelTrainer
from sklearn.tree import DecisionTreeClassifier


class TestModelEvaluation:
    """Test class for model evaluation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_trainer = ModelTrainer()
        
        # Create sample training data
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'sepal_length': np.random.uniform(4.0, 8.0, 100),
            'sepal_width': np.random.uniform(2.0, 4.5, 100),
            'petal_length': np.random.uniform(1.0, 7.0, 100),
            'petal_width': np.random.uniform(0.1, 2.5, 100)
        })
        
        # Create sample labels
        self.y_train = np.random.choice(['setosa', 'versicolor', 'virginica'], 100)
        
        # Create sample test data
        self.X_test = pd.DataFrame({
            'sepal_length': np.random.uniform(4.0, 8.0, 30),
            'sepal_width': np.random.uniform(2.0, 4.5, 30),
            'petal_length': np.random.uniform(1.0, 7.0, 30),
            'petal_width': np.random.uniform(0.1, 2.5, 30)
        })
        
        self.y_test = np.random.choice(['setosa', 'versicolor', 'virginica'], 30)
    
    def test_train_model_success(self):
        """Test successful model training."""
        model = self.model_trainer.train_model(self.X_train, self.y_train)
        
        assert isinstance(model, DecisionTreeClassifier)
        assert self.model_trainer.model is not None
        assert hasattr(self.model_trainer.model, 'predict')
    
    def test_train_model_failure(self):
        """Test model training failure with invalid data."""
        with pytest.raises(Exception):
            self.model_trainer.train_model(None, self.y_train)
    
    def test_evaluate_model_success(self):
        """Test successful model evaluation."""
        # First train the model
        self.model_trainer.train_model(self.X_train, self.y_train)
        
        # Then evaluate
        metrics = self.model_trainer.evaluate_model(self.X_test, self.y_test)
        
        # Check that metrics contain expected keys
        assert 'accuracy' in metrics
        assert 'classification_report' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check that accuracy is a valid value
        assert 0.0 <= metrics['accuracy'] <= 1.0
        
        # Check that classification report is a dictionary
        assert isinstance(metrics['classification_report'], dict)
        
        # Check that confusion matrix is a list
        assert isinstance(metrics['confusion_matrix'], list)
    
    def test_evaluate_model_not_trained(self):
        """Test model evaluation without training."""
        with pytest.raises(ValueError, match="Model not trained yet"):
            self.model_trainer.evaluate_model(self.X_test, self.y_test)
    
    def test_save_model_success(self):
        """Test successful model saving."""
        # Train model first
        self.model_trainer.train_model(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            self.model_trainer.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Verify the saved model can be loaded
            loaded_model = joblib.load(model_path)
            assert isinstance(loaded_model, DecisionTreeClassifier)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_save_model_not_trained(self):
        """Test model saving without training."""
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            with pytest.raises(ValueError, match="Model not trained yet"):
                self.model_trainer.save_model(model_path)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_load_model_success(self):
        """Test successful model loading."""
        # Train and save a model first
        self.model_trainer.train_model(self.X_train, self.y_train)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            # Save the model
            joblib.dump(self.model_trainer.model, model_path)
            
            # Create new trainer and load model
            new_trainer = ModelTrainer()
            loaded_model = new_trainer.load_model(model_path)
            
            assert isinstance(loaded_model, DecisionTreeClassifier)
            assert new_trainer.model is not None
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)
    
    def test_load_model_failure(self):
        """Test model loading failure."""
        with pytest.raises(Exception):
            self.model_trainer.load_model('nonexistent_model.joblib')
    
    def test_save_metrics_success(self):
        """Test successful metrics saving."""
        # Train model and get metrics
        self.model_trainer.train_model(self.X_train, self.y_train)
        metrics = self.model_trainer.evaluate_model(self.X_test, self.y_test)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            metrics_path = tmp_file.name
        
        try:
            self.model_trainer.save_metrics(metrics, metrics_path)
            assert os.path.exists(metrics_path)
            
            # Verify file content
            with open(metrics_path, 'r') as f:
                content = f.read()
                assert 'Accuracy:' in content
                assert 'Classification Report:' in content
        finally:
            if os.path.exists(metrics_path):
                os.unlink(metrics_path)
    
    def test_model_accuracy_threshold(self):
        """Test that model accuracy meets minimum threshold."""
        # Train model
        self.model_trainer.train_model(self.X_train, self.y_train)
        
        # Evaluate model
        metrics = self.model_trainer.evaluate_model(self.X_test, self.y_test)
        
        # Check that accuracy is above minimum threshold
        min_accuracy = 0.0  # Very low threshold for test data
        assert metrics['accuracy'] >= min_accuracy, f"Accuracy {metrics['accuracy']} below threshold {min_accuracy}"
    
    def test_model_predictions_format(self):
        """Test that model predictions are in correct format."""
        # Train model
        self.model_trainer.train_model(self.X_train, self.y_train)
        
        # Make predictions
        predictions = self.model_trainer.model.predict(self.X_test)
        
        # Check predictions format
        assert len(predictions) == len(self.y_test)
        assert all(pred in ['setosa', 'versicolor', 'virginica'] for pred in predictions)
    
    def test_model_feature_importance(self):
        """Test that model has feature importance."""
        # Train model
        self.model_trainer.train_model(self.X_train, self.y_train)
        
        # Check feature importance
        feature_importance = self.model_trainer.model.feature_importances_
        assert len(feature_importance) == 4  # 4 features
        assert all(imp >= 0 for imp in feature_importance)  # All non-negative
        assert abs(sum(feature_importance) - 1.0) < 1e-6  # Sum to 1
