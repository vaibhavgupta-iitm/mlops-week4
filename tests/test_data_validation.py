"""
Unit tests for data validation functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import DataProcessor


class TestDataValidation:
    """Test class for data validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.data_processor = DataProcessor()
        
        # Create valid test data
        self.valid_data = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7, 4.6, 5.0],
            'sepal_width': [3.5, 3.0, 3.2, 3.1, 3.6],
            'petal_length': [1.4, 1.4, 1.3, 1.5, 1.4],
            'petal_width': [0.2, 0.2, 0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa', 'setosa', 'setosa']
        })
        
        # Create invalid test data with missing columns
        self.invalid_data_missing_columns = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
        
        # Create invalid test data with null values
        self.invalid_data_nulls = pd.DataFrame({
            'sepal_length': [5.1, np.nan, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
        
        # Create invalid test data with wrong data types
        self.invalid_data_types = pd.DataFrame({
            'sepal_length': ['5.1', '4.9', '4.7'],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'setosa', 'setosa']
        })
        
        # Create invalid test data with unexpected species
        self.invalid_data_species = pd.DataFrame({
            'sepal_length': [5.1, 4.9, 4.7],
            'sepal_width': [3.5, 3.0, 3.2],
            'petal_length': [1.4, 1.4, 1.3],
            'petal_width': [0.2, 0.2, 0.2],
            'species': ['setosa', 'unknown', 'versicolor']
        })
    
    def test_validate_data_valid(self):
        """Test data validation with valid data."""
        result = self.data_processor.validate_data(self.valid_data)
        assert result is True
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        result = self.data_processor.validate_data(self.invalid_data_missing_columns)
        assert result is False
    
    def test_validate_data_nulls(self):
        """Test data validation with null values."""
        result = self.data_processor.validate_data(self.invalid_data_nulls)
        assert result is False
    
    def test_validate_data_wrong_types(self):
        """Test data validation with wrong data types."""
        result = self.data_processor.validate_data(self.invalid_data_types)
        assert result is False
    
    def test_validate_data_unexpected_species(self):
        """Test data validation with unexpected species values."""
        result = self.data_processor.validate_data(self.invalid_data_species)
        assert result is False
    
    def test_load_data_success(self):
        """Test successful data loading."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.valid_data
            result = self.data_processor.load_data('test_file.csv')
            assert result.equals(self.valid_data)
            mock_read_csv.assert_called_once_with('test_file.csv')
    
    def test_load_data_failure(self):
        """Test data loading failure."""
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = Exception("File not found")
            with pytest.raises(Exception):
                self.data_processor.load_data('nonexistent_file.csv')
    
    def test_split_data_success(self):
        """Test successful data splitting."""
        X_train, X_test, y_train, y_test = self.data_processor.split_data(self.valid_data)
        
        # Check that we get the expected number of features
        assert X_train.shape[1] == 4  # 4 feature columns
        assert X_test.shape[1] == 4
        
        # Check that train and test sets are not empty
        assert len(X_train) > 0
        assert len(X_test) > 0
        
        # Check that train and test sets together equal original data
        assert len(X_train) + len(X_test) == len(self.valid_data)
    
    def test_augment_data_success(self):
        """Test successful data augmentation."""
        original_size = len(self.valid_data)
        augmented_data = self.data_processor.augment_data(self.valid_data, n_extra_rows=5)
        
        # Check that augmented data has more rows
        assert len(augmented_data) > original_size
        assert len(augmented_data) == original_size + 5
        
        # Check that all original data is preserved
        assert augmented_data.iloc[:original_size].equals(self.valid_data)
    
    def test_augment_data_zero_rows(self):
        """Test data augmentation with zero extra rows."""
        augmented_data = self.data_processor.augment_data(self.valid_data, n_extra_rows=0)
        assert len(augmented_data) == len(self.valid_data)
        assert augmented_data.equals(self.valid_data)
