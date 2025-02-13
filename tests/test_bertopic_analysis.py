# tests/test_bertopic_analysis.py
import pytest
from src.bertopic_analysis import (
    calculate_topic_probabilities,
    rank_topics,
    find_category_in_topics
)

def test_calculate_topic_probabilities():
    test_cat_tweets = {
        'happiness': ([0, 1], [0.8, 0.9]),
        'sadness': ([0], [0.7])
    }
    
    result = calculate_topic_probabilities(test_cat_tweets)
    
    assert 'happiness' in result
    assert 'sadness' in result
    assert len(result['happiness']) == 2
    assert len(result['sadness']) == 1

def test_rank_topics():
    test_prob_array = {
        'happiness': {
            0: [2, 1.7],
            1: [1, 0.9]
        }
    }
    
    rank, ordered_rank = rank_topics(test_prob_array)
    
    assert 'happiness' in rank
    assert 'happiness' in ordered_rank
    assert len(ordered_rank['happiness']) == 2
    assert list(ordered_rank['happiness'].keys())[0] in [0, 1]

def test_find_category_in_topics(mocker):
    # Mock BERTopic model
    class MockBERTopic:
        def get_topic(self, topic_id):
            return [('happiness', 0.8)]
    
    test_ordered_rank = {
        'happiness': {0: 0.8, 1: 0.6}
    }
    
    test_model_list = {
        'happiness': MockBERTopic()
    }
    
    result = find_category_in_topics(test_ordered_rank, test_model_list)
    
    assert 'happiness' in result
    assert len(result['happiness']) > 0