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