# src/bertopic_analysis.py
from bertopic import BERTopic
import pandas as pd
from typing import Dict, List, Tuple, Any

def perform_bertopic_analysis(df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """
    Perform BERTopic analysis on tweets grouped by sentiment category
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Sentiment Category' and 'Tweets' columns
    
    Returns:
    --------
    Tuple[Dict, Dict]
        cat_tweets: Dictionary containing topics and probabilities for each category
        model_list: Dictionary containing BERTopic models for each category
    """
    cat_tweets = {}
    model_list = {}
    
    for index, row in df.iterrows():
        sentiment_category = row['Sentiment Category']
        tweets = row['Tweets']
        
        topic_model = BERTopic()
        topics, probs = topic_model.fit_transform(tweets)
        
        cat_tweets[sentiment_category] = [topics, probs]
        model_list[sentiment_category] = topic_model
    
    return cat_tweets, model_list

def calculate_topic_probabilities(cat_tweets: Dict) -> Dict:
    """
    Calculate probability array for topics in each category
    
    Parameters:
    -----------
    cat_tweets : Dict
        Dictionary containing topics and probabilities for each category
    
    Returns:
    --------
    Dict
        Dictionary containing probability calculations for each topic
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

def rank_topics(probability_array: Dict) -> Tuple[Dict, Dict]:
    """
    Rank topics based on their probabilities and tweet counts
    
    Parameters:
    -----------
    probability_array : Dict
        Dictionary containing probability calculations for each topic
    
    Returns:
    --------
    Tuple[Dict, Dict]
        tot_tweet: Dictionary containing total tweet counts per category
        ordered_rank: Dictionary containing ranked topics per category
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
            rank[sentiment_category][topic] = perc_tweets*perc_prob*100
    
    ordered_rank = {}
    for sentiment_cat in rank:
        ordered_rank[sentiment_cat] = dict(sorted(rank[sentiment_cat].items(), 
                                                key=lambda item: item[1], 
                                                reverse=True))
    
    return tot_tweet, ordered_rank

def analyze_category_in_topics(ordered_rank: Dict, model_list: Dict) -> Dict:
    """
    Analyze if category title appears in topics
    
    Parameters:
    -----------
    ordered_rank : Dict
        Dictionary containing ranked topics per category
    model_list : Dict
        Dictionary containing BERTopic models for each category
    
    Returns:
    --------
    Dict
        Dictionary containing topics where category appears
    """
    category_in_topic = {}
    
    for sentiment_cat in ordered_rank:
        category_in_topic[sentiment_cat] = []
        for topic, prob in ordered_rank[sentiment_cat].items():
            if any(sentiment_cat in tupla for tupla in model_list[sentiment_cat].get_topic(topic)):
                category_in_topic[sentiment_cat].append([topic, prob])
    
    return category_in_topic

def visualize_topics_for_category(model_list: Dict, category: str = None):
    """
    Visualize topics for a specific category or all categories
    
    Parameters:
    -----------
    model_list : Dict
        Dictionary containing BERTopic models for each category
    category : str, optional
        Specific category to visualize. If None, visualizes all categories
    """
    if category:
        try:
            model_list[category].visualize_topics()
        except Exception as e:
            print(f"{category} too few elements")
    else:
        for sentiment_cat in model_list:
            try:
                model_list[sentiment_cat].visualize_topics()
            except Exception as e:
                print(f"{sentiment_cat} too few elements")