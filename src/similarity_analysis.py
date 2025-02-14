import pandas as pd
from fuzzywuzzy import fuzz
from typing import List, Dict

def calculate_ratio_scores(strings_list: List[str]) -> List[float]:
    """
    Calculate similarity ratios between all pairs of strings in the input list.
    
    Args:
        strings_list (List[str]): List of strings to compare
        
    Returns:
        List[float]: List of similarity ratios between all pairs
    """
    ratio_scores = []
    for i in range(len(strings_list)):
        for j in range(i + 1, len(strings_list)):
            ratio = fuzz.ratio(strings_list[i], strings_list[j])
            ratio_scores.append(ratio)
    return ratio_scores

def analyze_tweet_similarities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze similarities between tweets for each sentiment category.
    
    Args:
        df (pd.DataFrame): DataFrame containing tweets by category.
                          Expected columns: sentiment category in first column,
                          'Tweets' column containing lists of tweets
        
    Returns:
        pd.DataFrame: DataFrame with similarity statistics per category
    """
    result_data = []
    ratios = {}
    
    for i in range(len(df)):
        sentiment_category = df.iloc[i, 0]
        tweets = df.Tweets[i]
        
        ratio_scores = calculate_ratio_scores(tweets)
        
        if ratio_scores:
            ratios[sentiment_category] = ratio_scores
            result_data.append({
                'Sentiment Category': sentiment_category,
                'Min Ratio Score': min(ratio_scores),
                'Max Ratio Score': max(ratio_scores),
                'Average Ratio Score': sum(ratio_scores) / len(ratio_scores)
            })
    
    return pd.DataFrame(result_data)