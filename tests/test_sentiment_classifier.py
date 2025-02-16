import pytest
import numpy as np
import pandas as pd
from src.sentiment_classifier import SentimentClassifier

# File: tests/test_sentiment_classifier.py

@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing with sufficient samples per class."""
    return pd.DataFrame({
        'content': [
            'This is great!',
            'I am so happy',
            'This is awesome',
            'Wonderful day',
            'This is terrible',
            'I am very sad',
            'This is awful',
            'Really disappointed',
            'I am angry',
            'This makes me mad',
            'Furious about this',
            'So angry right now'
        ],
        'sentiment': [
            'positive',
            'positive',
            'positive',
            'positive',
            'negative',
            'negative',
            'negative',
            'negative',
            'angry',
            'angry',
            'angry',
            'angry'
        ]
    })

@pytest.fixture
def classifier():
    """Fixture providing a classifier instance."""
    return SentimentClassifier(max_features=100)

def test_invalid_test_size():
    """Test handling of invalid test_size values."""
    with pytest.raises(ValueError, match="test_size must be a float between 0 and 1"):
        SentimentClassifier(test_size=1.5)
    with pytest.raises(ValueError, match="test_size must be a float between 0 and 1"):
        SentimentClassifier(test_size=0)
    with pytest.raises(ValueError, match="test_size must be a float between 0 and 1"):
        SentimentClassifier(test_size=-0.1)

def test_insufficient_samples():
    """Test handling of insufficient samples per class."""
    small_data = pd.DataFrame({
        'content': ['This is great!', 'I am angry', 'This is bad'],
        'sentiment': ['positive', 'angry', 'negative']
    })
    classifier = SentimentClassifier(test_size=0.25)
    with pytest.raises(ValueError, match="Insufficient samples per class"):
        classifier.prepare_data(small_data)

def test_data_preparation(classifier, sample_data):
    """Test the data preparation method."""
    x_train, x_test, y_train, y_test, labels, mapping = classifier.prepare_data(sample_data)
    
    # Check types
    assert isinstance(x_train, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(labels, np.ndarray)
    
    # Check dimensions
    assert x_train.shape[1] <= classifier.max_features
    assert x_test.shape[1] <= classifier.max_features
    assert len(y_train.shape) == 1
    assert len(y_test.shape) == 1
    
    # Check class distribution
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    assert len(unique_train) == 3  # positive, negative, angry
    assert all(counts_train >= 1)  # At least one sample per class in training

@pytest.mark.parametrize("max_features", [10, 50, 100])
def test_different_feature_sizes(sample_data, max_features):
    """Test classifier with different maximum feature sizes."""
    classifier = SentimentClassifier(max_features=max_features)
    x_train, x_test, y_train, y_test, labels, _ = classifier.prepare_data(sample_data)
    
    # Check feature dimensions
    assert x_train.shape[1] <= max_features
    assert x_test.shape[1] <= max_features

def test_empty_input():
    """Test handling of empty input data."""
    classifier = SentimentClassifier()
    empty_data = pd.DataFrame(columns=['content', 'sentiment'])
    with pytest.raises(ValueError):
        classifier.prepare_data(empty_data)

def test_invalid_input_data():
    """Test handling of invalid input data."""
    classifier = SentimentClassifier()
    
    # Test missing columns
    invalid_data = pd.DataFrame({'wrong_column': ['text']})
    with pytest.raises(ValueError):
        classifier.prepare_data(invalid_data)

def test_small_class_handling(classifier):
    """Test handling of data with small classes."""
    small_data = pd.DataFrame({
        'content': ['This is great!', 'I am angry'],
        'sentiment': ['positive', 'angry']
    })
    
    # Should raise ValueError due to insufficient samples per class
    with pytest.raises(ValueError, match="Insufficient samples per class"):
        classifier.prepare_data(small_data)

def test_model_training_workflow(classifier, sample_data):
    """Test the complete model training workflow."""
    # Prepare data
    x_train, x_test, y_train, y_test, labels, _ = classifier.prepare_data(sample_data)
    
    # Train Random Forest
    rf_accuracy, rf_params, rf_proba = classifier.train_random_forest(
        x_train, x_test, y_train, y_test, labels
    )
    assert 0 <= rf_accuracy <= 1
    assert isinstance(rf_params, dict)
    assert rf_proba.shape[0] == len(y_test)
    assert rf_proba.shape[1] == len(labels)
    
    # Train XGBoost
    xgb_accuracy, xgb_params, xgb_proba = classifier.train_xgboost(
        x_train, x_test, y_train, y_test, labels
    )
    assert 0 <= xgb_accuracy <= 1
    assert isinstance(xgb_params, dict)
    assert xgb_proba.shape[0] == len(y_test)
    assert xgb_proba.shape[1] == len(labels)

def test_predict_method(classifier, sample_data):
    """Test the predict method."""
    # Train the model first
    x_train, x_test, y_train, y_test, labels, _ = classifier.prepare_data(sample_data)
    classifier.train_random_forest(x_train, x_test, y_train, y_test, labels)
    
    # Test predictions
    new_texts = ["This is great!", "I am very sad"]
    predictions, probabilities = classifier.predict(new_texts)
    
    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)
    assert len(predictions) == len(new_texts)
    assert probabilities.shape == (len(new_texts), len(labels))
    assert all(0 <= prob <= 1 for prob in probabilities.flatten())