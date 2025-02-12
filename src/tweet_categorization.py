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

# src/tweet_categorization.py

def analyze_empath_categories(csv_path):
   """
   Analyze tweets using Empath and calculate statistics for each category
   
   Args:
       csv_path: Path to CSV file containing concatenated tweets
       
   Returns:
       tuple: (DataFrame with analysis, media dictionary, positive values dictionary)
   """
   lexicon = Empath()
   concatenated_df = pd.read_csv(csv_path)
   concatenated_df['Key Categories'] = ""
   media = {}
   positive_values_of_cat = {}
   
   for index, row in concatenated_df.iterrows():
       concatenated_tweets = row['Concatenated Tweets']
       sentiment_category = row['Sentiment Category'].strip().lower()
       empath_categories = lexicon.analyze(concatenated_tweets, normalize=True)
       
       # Get positive values and calculate mean
       positive_values = [value for category, value in empath_categories.items() if value > 0]
       positive_values_of_cat[sentiment_category] = positive_values
       media[sentiment_category] = sum(positive_values) / len(positive_values)
       
       # Sort values for plotting
       positive_values_of_cat[sentiment_category].sort()
       
   return concatenated_df, media, positive_values_of_cat


# src/tweet_categorization.py
# ... (codice precedente) ...

def get_key_categories(concatenated_df, media):
    """Extract key categories based on Empath analysis"""
    lexicon = Empath()
    
    for index, row in concatenated_df.iterrows():
        concatenated_tweets = row['Concatenated Tweets']
        sentiment_category = row['Sentiment Category'].strip().lower()
        empath_categories = lexicon.analyze(concatenated_tweets, normalize=True)
        non_zero_categories = [category for category, value in empath_categories.items() 
                             if value > media[sentiment_category]]
        concatenated_df.at[index, 'Key Categories'] = ' '.join(non_zero_categories)
    
    return concatenated_df

def analyze_wordnet_relationships(concatenated_df):
    """Analyze relationships between categories using WordNet"""
    results = []
    
    for index, row in concatenated_df.iterrows():
        empath_categories = row['Key Categories'].split()
        sentiment_category = row['Sentiment Category'].strip().lower()
        flag = sentiment_category in empath_categories
        hyponyms = []
        hypernyms = []
        
        for keyword in empath_categories:
            synsets_keyword = wordnet.synsets(keyword)
            synsets_category = wordnet.synsets(sentiment_category)
            
            for synset_keyword in synsets_keyword:
                for synset_category in synsets_category:
                    if synset_keyword.hyponyms() and synset_category in synset_keyword.hyponyms():
                        hyponyms.append(keyword)
                    elif synset_keyword.hypernyms() and synset_category in synset_keyword.hypernyms():
                        hypernyms.append(keyword)
        
        result = {
            'Sentiment Category': sentiment_category,
            'Same Category': flag,
            'Hyponyms': ', '.join(hyponyms),
            'Hypernyms': ', '.join(hypernyms)
        }
        results.append(result)
    
    return pd.DataFrame(results)