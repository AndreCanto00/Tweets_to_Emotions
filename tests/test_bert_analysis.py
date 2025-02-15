import pytest
import pandas as pd
import numpy as np
from src.bert_analysis import BertSemanticAnalyzer, analyze_semantic_similarities

@pytest.fixture
def sample_data():
    # Create sample DataFrame
    tweets_df = pd.DataFrame({
        'Sentiment Category': ['happy', 'sad'],
        'Tweets': [
            ['I am very happy today', 'What a great day'],
            ['Feeling blue', 'Not a good day']
        ]
    })
    
    # Create sample antonyms dictionary
    antonyms_dict = {
        'happy': ['sad', 'unhappy'],
        'sad': ['happy', 'joyful']
    }
    
    return tweets_df, antonyms_dict

@pytest.fixture
def bert_analyzer():
    return BertSemanticAnalyzer()

def test_bert_embedding(bert_analyzer):
    embedding = bert_analyzer.get_bert_embedding("test text")
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1  # Should be a 1D vector
    
def test_calculate_similarities(bert_analyzer, sample_data):
    tweets_df, antonyms_dict = sample_data
    similarities = bert_analyzer.calculate_tweet_antonym_similarities(
        tweets_df, antonyms_dict
    )
    
    # Check structure
    assert isinstance(similarities, dict)
    assert all(isinstance(v, dict) for v in similarities.values())
    
    # Check values
    for category_dict in similarities.values():
        for score in category_dict.values():
            assert isinstance(score, (float, np.float32, np.float64))
            assert 0 <= score <= 1

def test_summarize_similarities(bert_analyzer):
    test_similarities = {
        'happy': {0: 0.8, 1: 0.9},
        'sad': {0: 0.7, 1: 0.6}
    }
    
    summary = bert_analyzer.summarize_similarities(test_similarities)
    
    assert isinstance(summary, pd.DataFrame)
    assert set(summary.columns) == {'Sentiment Category', 'Min', 'Max', 'Mean'}
    assert len(summary) == 2
    
    # Check calculations
    happy_row = summary[summary['Sentiment Category'] == 'happy'].iloc[0]
    assert happy_row['Min'] == 0.8
    assert happy_row['Max'] == 0.9
    assert happy_row['Mean'] == pytest.approx(0.85)

def test_full_analysis(sample_data):
    tweets_df, antonyms_dict = sample_data
    similarities, summary = analyze_semantic_similarities(
        tweets_df, antonyms_dict
    )
    
    assert isinstance(similarities, dict)
    assert isinstance(summary, pd.DataFrame)
    
    # Check that all categories are present
    assert set(similarities.keys()) == set(antonyms_dict.keys())
    assert set(summary['Sentiment Category']) == set(antonyms_dict.keys())

def test_empty_inputs():
    empty_df = pd.DataFrame(columns=['Sentiment Category', 'Tweets'])
    empty_dict = {}
    
    similarities, summary = analyze_semantic_similarities(
        empty_df, empty_dict
    )
    
    assert isinstance(similarities, dict)
    assert isinstance(summary, pd.DataFrame)
    assert len(similarities) == 0
    assert len(summary) == 0

@pytest.mark.parametrize("text", [
    "Simple text",
    "Longer text with multiple words",
    "",  # Empty string
    "Special !@#$ characters"
])
def test_bert_embedding_various_inputs(bert_analyzer, text):
    embedding = bert_analyzer.get_bert_embedding(text)
    assert isinstance(embedding, np.ndarray)