# src/tweet_categorization.py
import pandas as pd
from empath import Empath
from nltk.corpus import wordnet

def create_category_tweets(data):
    """Create dictionaries of tweets grouped by sentiment category"""
    category_tweets = {}
    for index, row in data.iterrows():
        sentiment_category = row['sentiment'].strip().lower()
        tweet_text = row['content']
        if sentiment_category in category_tweets:
            category_tweets[sentiment_category].append(tweet_text)
        else:
            category_tweets[sentiment_category] = [tweet_text]
    return category_tweets

def save_categorized_tweets(category_tweets, concatenated=True):
    """Save tweets either concatenated or detached"""
    if concatenated:
        df = pd.DataFrame({
            'Sentiment Category': list(category_tweets.keys()),
            'Concatenated Tweets': ['\n'.join(category_tweets[cat]) for cat in category_tweets]
        })
        filename = "concatenated_tweets_by_category.csv"
    else:
        df = pd.DataFrame({
            'Sentiment Category': list(category_tweets.keys()),
            'Tweets': [category_tweets[cat] for cat in category_tweets]
        })
        filename = "detached_tweets_by_category.csv"
    
    df.to_csv(filename, index=False)
    return df

def analyze_with_empath(concatenated_df):
    """Analyze tweets using Empath lexicon"""
    lexicon = Empath()
    media = {}
    positive_values_of_cat = {}
    
    for index, row in concatenated_df.iterrows():
        concatenated_tweets = row['Concatenated Tweets']
        sentiment_category = row['Sentiment Category'].strip().lower()
        empath_categories = lexicon.analyze(concatenated_tweets, normalize=True)
        positive_values = [value for _, value in empath_categories.items() if value > 0]
        positive_values_of_cat[sentiment_category] = positive_values
        media[sentiment_category] = sum(positive_values) / len(positive_values)
    
    return positive_values_of_cat, media