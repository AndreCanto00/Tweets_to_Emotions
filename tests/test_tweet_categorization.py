import pytest
import pandas as pd
from src.tweet_categorization import create_category_tweets, save_categorized_tweets

def test_create_category_tweets():
    test_data = pd.DataFrame({
        'content': ['happy tweet', 'sad tweet', 'another happy'],
        'sentiment': ['happiness', 'sadness', 'happiness']
    })
    result = create_category_tweets(test_data)
    
    assert 'happiness' in result
    assert 'sadness' in result
    assert len(result['happiness']) == 2
    assert len(result['sadness']) == 1

def test_save_categorized_tweets():
    test_categories = {
        'happiness': ['tweet1', 'tweet2'],
        'sadness': ['tweet3']
    }
    
    # Test concatenated
    concat_df = save_categorized_tweets(test_categories, concatenated=True)
    assert 'Concatenated Tweets' in concat_df.columns
    assert len(concat_df) == 2
    
    # Test detached
    detached_df = save_categorized_tweets(test_categories, concatenated=False)
    assert 'Tweets' in detached_df.columns
    assert len(detached_df) == 2

    # tests/test_tweet_categorization.py

def test_analyze_empath_categories(tmp_path):
    """Test Empath analysis function with sample data"""
    # Create temporary test CSV
    test_df = pd.DataFrame({
        'Sentiment Category': ['happiness', 'sadness'],
        'Concatenated Tweets': ['happy joyful tweet', 'sad lonely tweet']
    })
    csv_path = tmp_path / "test_tweets.csv"
    test_df.to_csv(csv_path, index=False)
    
    # Test function
    df, media, positive_values = analyze_empath_categories(csv_path)
    
    assert isinstance(df, pd.DataFrame)
    assert 'Key Categories' in df.columns
    assert len(media) == 2
    assert len(positive_values) == 2
    assert all(isinstance(v, list) for v in positive_values.values())


# tests/test_tweet_categorization.py
# ... (test precedenti) ...

def test_get_key_categories():
    test_df = pd.DataFrame({
        'Sentiment Category': ['happiness', 'sadness'],
        'Concatenated Tweets': ['happy joyful tweet', 'sad lonely tweet']
    })
    test_media = {'happiness': 0.5, 'sadness': 0.5}
    
    result = get_key_categories(test_df, test_media)
    assert 'Key Categories' in result.columns
    assert isinstance(result['Key Categories'][0], str)

def test_analyze_wordnet_relationships():
    test_df = pd.DataFrame({
        'Sentiment Category': ['happiness'],
        'Key Categories': ['joy happiness pleasure']
    })
    
    result = analyze_wordnet_relationships(test_df)
    assert all(col in result.columns for col in ['Sentiment Category', 'Same Category', 'Hyponyms', 'Hypernyms'])
    assert len(result) == 1