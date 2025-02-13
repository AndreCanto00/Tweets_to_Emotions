# src/bertopic_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def calculate_topic_probabilities(cat_tweets: Dict[str, Tuple[List, List]]) -> Dict[str, Dict[int, List]]:
    """
    Calculate probability arrays for each sentiment category and topic
    
    Parameters:
    -----------
    cat_tweets : dict
        Dictionary containing sentiment categories with their topics and probabilities
    
    Returns:
    --------
    dict
        Nested dictionary with probability arrays for each category and topic
    """
    probability_array = {}
    
    for sentiment_category, values in cat_tweets.items():
        topics, probs = values
        probability_array[sentiment_category] = {}
        
        for topic, prob in zip(topics, probs):
            if topic not in probability_array[sentiment_category]:
                probability_array[sentiment_category][topic] = [0, 0]
            pre_topic, pre_prob = probability_array[sentiment_category][topic]
            probability_array[sentiment_category][topic] = [pre_topic + 1, pre_prob + prob]
    
    return probability_array

def rank_topics(probability_array: Dict[str, Dict[int, List]]) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    """
    Rank topics based on their probabilities and tweet counts
    
    Parameters:
    -----------
    probability_array : dict
        Nested dictionary with probability arrays
    
    Returns:
    --------
    tuple
        Contains two dictionaries:
        - Ranking of topics
        - Ordered ranking of topics
    """
    tot_tweet = {}
    rank = {}
    
    for sentiment_category, i in probability_array.items():
        tot_tweet[sentiment_category] = sum([valore[0] for valore in probability_array[sentiment_category].values()])
        rank[sentiment_category] = {}
        
        for topic in probability_array[sentiment_category]:
            tweet, prob = probability_array[sentiment_category][topic]
            perc_tweets = tweet/tot_tweet[sentiment_category]
            perc_prob = prob/tweet
            rank[sentiment_category][topic] = perc_tweets * perc_prob * 100
    
    # Order rankings
    ordered_rank = {}
    for sentiment_cat in rank:
        ordered_rank[sentiment_cat] = dict(sorted(
            rank[sentiment_cat].items(), 
            key=lambda item: item[1], 
            reverse=True
        ))
    
    return rank, ordered_rank

def find_category_in_topics(ordered_rank: Dict[str, Dict[int, float]], 
                          model_list: Dict[str, object]) -> Dict[str, List]:
    """
    Find if category title appears in topics
    
    Parameters:
    -----------
    ordered_rank : dict
        Ordered ranking of topics
    model_list : dict
        Dictionary of BERTopic models for each category
    
    Returns:
    --------
    dict
        Dictionary containing topics where category appears
    """
    category_in_topic = {}
    
    for sentiment_cat in ordered_rank:
        category_in_topic[sentiment_cat] = []
        for topic, prob in ordered_rank[sentiment_cat].items():
            if any(sentiment_cat in tupla 
                  for tupla in model_list[sentiment_cat].get_topic(topic)):
                category_in_topic[sentiment_cat].append([topic, prob])
    
    return category_in_topic

def visualize_topic_models(model_list: Dict[str, object]) -> None:
    """
    Visualize topics for each model
    
    Parameters:
    -----------
    model_list : dict
        Dictionary of BERTopic models for each category
    """
    for sentiment_cat in model_list:
        try:
            model_list[sentiment_cat].visualize_topics()
        except Exception as e:
            print(f"{sentiment_cat} has too few elements")