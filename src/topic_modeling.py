# src/topic_modeling.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def perform_lda_analysis(df, n_topics=1, n_words_per_topic=3):
   """
   Perform LDA topic modeling on tweets
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing 'Sentiment Category' and 'Concatenated Tweets' columns
   n_topics : int, optional (default=1)
       Number of topics for LDA
   n_words_per_topic : int, optional (default=3)
       Number of keywords per topic to extract
   
   Returns:
   --------
   pandas.DataFrame
       DataFrame containing sentiment categories and their LDA keywords
   """
   results = []
   
   for index, row in df.iterrows():
       sentiment_category = row['Sentiment Category']
       concatenated_tweets = row['Concatenated Tweets']
       
       # Create and fit vectorizer
       vectorizer = CountVectorizer()
       X = vectorizer.fit_transform([concatenated_tweets])
       
       # Perform LDA
       lda = LatentDirichletAllocation(
           n_components=n_topics,
           random_state=42
       )
       lda.fit(X)
       
       # Get top keywords
       feature_names = vectorizer.get_feature_names_out()
       topic_keywords = []
       for topic_idx, topic in enumerate(lda.components_):
           top_keywords = [feature_names[i] 
                         for i in topic.argsort()[:-n_words_per_topic-1:-1]]
           topic_keywords.extend(top_keywords)
       
       result = {
           'Sentiment Category': sentiment_category,
           'LDA Keywords': topic_keywords
       }
       results.append(result)
   
   return pd.DataFrame(results)

def analyze_keyword_matches(results_df):
   """
   Analyze matches between LDA keywords and sentiment categories
   
   Parameters:
   -----------
   results_df : pandas.DataFrame
       DataFrame containing 'Sentiment Category' and 'LDA Keywords' columns
   
   Returns:
   --------
   pandas.DataFrame
       DataFrame containing analysis of string and semantic matches
   """
   analysis_results = []
   
   for _, row in results_df.iterrows():
       sentiment_category = row['Sentiment Category'].strip().lower()
       keywords = row['LDA Keywords']
       
       # Check for exact string match
       string_match = any(keyword.lower() == sentiment_category 
                        for keyword in keywords)
       
       # Check for semantic relationships
       hyponyms = []
       hypernyms = []
       
       for keyword in keywords:
           synsets_keyword = wordnet.synsets(keyword)
           synsets_category = wordnet.synsets(sentiment_category)
           
           for synset_keyword in synsets_keyword:
               for synset_category in synsets_category:
                   if synset_keyword.hyponyms() and synset_category in synset_keyword.hyponyms():
                       hyponyms.append(keyword)
                   elif synset_keyword.hypernyms() and synset_category in synset_keyword.hypernyms():
                       hypernyms.append(keyword)
       
       analysis_results.append({
           'Sentiment Category': sentiment_category,
           'LDA Keywords': ', '.join(keywords),
           'String Match': string_match,
           'Hyponyms': ', '.join(hyponyms),
           'Hypernyms': ', '.join(hypernyms)
       })
   
   return pd.DataFrame(analysis_results)

def analyze_lda_relationships(df, n_topics=1, n_words_per_topic=3):
   """
   Perform LDA analysis and analyze relationships between keywords and categories
   
   Parameters:
   -----------
   df : pandas.DataFrame
       DataFrame containing 'Sentiment Category' and 'Concatenated Tweets' columns
   n_topics : int, optional (default=1)
       Number of topics for LDA
   n_words_per_topic : int, optional (default=3)
       Number of keywords per topic to extract
   
   Returns:
   --------
   pandas.DataFrame
       DataFrame containing LDA results and relationship analysis
   """
   wnl = WordNetLemmatizer()
   lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
   results = []
   
   for index, row in df.iterrows():
       sentiment_category = row['Sentiment Category']
       concatenated_tweets = row['Concatenated Tweets']
       
       # Perform LDA analysis
       vectorizer = CountVectorizer()
       X = vectorizer.fit_transform([concatenated_tweets])
       lda.fit(X)
       
       # Get top keywords
       topic_keywords = [vectorizer.get_feature_names_out()[i] 
                        for i in lda.components_.argsort()[:, -n_words_per_topic:]]
       
       # Check for exact category match
       is_same_category = sentiment_category in topic_keywords[0]
       
       # Analyze WordNet relationships
       hyponyms = []
       hypernyms = []
       
       for keyword in topic_keywords[0]:
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
           'LDA Keywords': topic_keywords[0],
           'Same Category': is_same_category,
           'Hyponyms': ', '.join(hyponyms),
           'Hypernyms': ', '.join(hypernyms)
       }
       results.append(result)
   
   return pd.DataFrame(results)

def save_results_to_csv(results_df, filename):
   """
   Save analysis results to CSV file
   
   Parameters:
   -----------
   results_df : pandas.DataFrame
       DataFrame containing analysis results
   filename : str
       Name of the file to save results to
   """
   results_df.to_csv(filename, index=False)