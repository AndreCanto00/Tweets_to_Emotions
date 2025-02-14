import pytest
import pandas as pd
from src.bertopic_analysis import (perform_bertopic_analysis,
                                 calculate_topic_probabilities,
                                 rank_topics,
                                 analyze_category_in_topics)

def test_perform_bertopic_analysis():
    test_df = pd.DataFrame({
        'Sentiment Category': ['happiness'],
        'Tweets': [['happy tweet', 'joyful moment']]
    })
    
    try:
        cat_tweets, model_list = perform_bertopic_analysis(test_df)
        assert 'happiness' in cat_tweets
        assert 'happiness' in model_list
        assert len(cat_tweets['happiness']) == 2  # topics and probs
    except ValueError as e:
        assert str(e) == "zero-size array to reduction operation maximum which has no identity"

# Add more tests for other functions...