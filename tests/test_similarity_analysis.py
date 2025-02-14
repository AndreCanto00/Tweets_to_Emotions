import pytest
import pandas as pd
from src.similarity_analysis import calculate_ratio_scores, analyze_tweet_similarities

@pytest.fixture
def sample_strings():
    return [
        "Hello world",
        "Hello World!",
        "Something completely different"
    ]

@pytest.fixture
def sample_tweets_df():
    return pd.DataFrame({
        'Category': ['Happy', 'Sad'],
        'Tweets': [
            ['I am happy!', 'This is great!', 'Wonderful day'],
            ['Feeling down', 'Not a good day', 'So sad']
        ]
    })

def test_calculate_ratio_scores(sample_strings):
    scores = calculate_ratio_scores(sample_strings)
    
    # Verifica tipo e range dei punteggi
    assert all(isinstance(score, (int, float)) for score in scores)
    assert all(0 <= score <= 100 for score in scores)
    
    # Verifica numero corretto di confronti
    expected_comparisons = (len(sample_strings) * (len(sample_strings) - 1)) // 2
    assert len(scores) == expected_comparisons

def test_analyze_tweet_similarities(sample_tweets_df):
    result_df = analyze_tweet_similarities(sample_tweets_df)
    
    # Verifica colonne
    expected_columns = {'Sentiment Category', 'Min Ratio Score', 
                       'Max Ratio Score', 'Average Ratio Score'}
    assert set(result_df.columns) == expected_columns
    
    # Verifica numero righe
    assert len(result_df) == len(sample_tweets_df)
    
    # Verifica range dei punteggi
    assert all(result_df['Min Ratio Score'] >= 0)
    assert all(result_df['Max Ratio Score'] <= 100)
    assert all(result_df['Average Ratio Score'] >= result_df['Min Ratio Score'])
    assert all(result_df['Average Ratio Score'] <= result_df['Max Ratio Score'])

def test_empty_input():
    assert calculate_ratio_scores([]) == []
    
    empty_df = pd.DataFrame(columns=['Category', 'Tweets'])
    result_df = analyze_tweet_similarities(empty_df)
    assert result_df.empty

def test_single_tweet():
    single_tweet_df = pd.DataFrame({
        'Category': ['Happy'],
        'Tweets': [['Single tweet']]
    })
    result_df = analyze_tweet_similarities(single_tweet_df)
    assert result_df.empty  # Non ci dovrebbero essere confronti possibili