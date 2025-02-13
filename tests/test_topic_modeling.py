# tests/test_topic_modeling.py
import pytest
import pandas as pd
import numpy as np
from src.topic_modeling import (
   perform_lda_analysis, 
   analyze_keyword_matches,
   analyze_lda_relationships,
   save_results_to_csv
)
import os

@pytest.fixture
def sample_tweets_df():
   """Fixture to create a sample DataFrame for testing"""
   return pd.DataFrame({
       'Sentiment Category': ['happiness', 'sadness', 'anger'],
       'Concatenated Tweets': [
           'happy joyful smile wonderful great positive',
           'sad crying tears painful bad negative',
           'angry furious mad rage hate upset'
       ]
   })

@pytest.fixture
def sample_lda_results():
   """Fixture to create sample LDA results for testing"""
   return pd.DataFrame({
       'Sentiment Category': ['happiness', 'sadness'],
       'LDA Keywords': [
           ['happy', 'joy', 'smile'],
           ['sad', 'crying', 'tears']
       ]
   })

def test_perform_lda_analysis(sample_tweets_df):
   """Test basic LDA analysis functionality"""
   result = perform_lda_analysis(sample_tweets_df)
   
   # Check basic structure
   assert isinstance(result, pd.DataFrame)
   assert len(result) == 3
   assert all(col in result.columns for col in ['Sentiment Category', 'LDA Keywords'])
   
   # Check content types
   assert isinstance(result['LDA Keywords'].iloc[0], list)
   assert len(result['LDA Keywords'].iloc[0]) == 3  # default n_words_per_topic
   
   # Check that keywords are strings
   for keywords in result['LDA Keywords']:
       assert all(isinstance(word, str) for word in keywords)

def test_analyze_keyword_matches(sample_lda_results):
   """Test keyword matching analysis"""
   result = analyze_keyword_matches(sample_lda_results)
   
   # Check structure
   assert isinstance(result, pd.DataFrame)
   assert len(result) == 2
   assert all(col in result.columns for col in [
       'Sentiment Category',
       'LDA Keywords',
       'String Match',
       'Hyponyms',
       'Hypernyms'
   ])
   
   # Check data types
   assert isinstance(result['String Match'].iloc[0], bool)
   assert isinstance(result['Hyponyms'].iloc[0], str)
   assert isinstance(result['Hypernyms'].iloc[0], str)

def test_analyze_lda_relationships(sample_tweets_df):
   """Test combined LDA and relationship analysis"""
   result = analyze_lda_relationships(sample_tweets_df)
   
   # Check structure
   assert isinstance(result, pd.DataFrame)
   assert len(result) == 3
   assert all(col in result.columns for col in [
       'Sentiment Category',
       'LDA Keywords',
       'Same Category',
       'Hyponyms',
       'Hypernyms'
   ])
   
   # Check data types
   assert isinstance(result['Same Category'].iloc[0], bool)
   assert isinstance(result['Hyponyms'].iloc[0], str)
   assert isinstance(result['Hypernyms'].iloc[0], str)
   assert isinstance(result['LDA Keywords'].iloc[0], list)

def test_analyze_lda_relationships_empty_data():
   """Test behavior with empty DataFrame"""
   empty_df = pd.DataFrame({
       'Sentiment Category': [],
       'Concatenated Tweets': []
   })
   
   result = analyze_lda_relationships(empty_df)
   assert len(result) == 0
   assert isinstance(result, pd.DataFrame)

def test_perform_lda_analysis_custom_params(sample_tweets_df):
   """Test LDA analysis with custom parameters"""
   n_topics = 2
   n_words = 5
   result = perform_lda_analysis(sample_tweets_df, n_topics=n_topics, n_words_per_topic=n_words)
   
   assert len(result['LDA Keywords'].iloc[0]) == n_topics * n_words

def test_save_results_to_csv(sample_lda_results, tmp_path):
   """Test saving results to CSV"""
   # Create temporary file path
   temp_file = tmp_path / "test_results.csv"
   
   # Save results
   save_results_to_csv(sample_lda_results, temp_file)
   
   # Check if file exists
   assert os.path.exists(temp_file)
   
   # Read and verify contents
   loaded_df = pd.read_csv(temp_file)
   assert len(loaded_df) == len(sample_lda_results)
   assert all(col in loaded_df.columns for col in sample_lda_results.columns)

def test_analyze_lda_relationships_invalid_input():
   """Test behavior with invalid input"""
   invalid_df = pd.DataFrame({
       'Wrong Column': ['test']
   })
   
   with pytest.raises(KeyError):
       analyze_lda_relationships(invalid_df)

def test_analyze_keyword_matches_edge_cases(sample_lda_results):
   """Test keyword matching with edge cases"""
   # Add edge case with empty keywords
   edge_case_df = sample_lda_results.copy()
   edge_case_df.loc[len(edge_case_df)] = ['test', []]
   
   result = analyze_keyword_matches(edge_case_df)
   assert len(result) == len(edge_case_df)
   assert result['String Match'].iloc[-1] == False

def test_perform_lda_analysis_with_preprocessing(sample_tweets_df):
   """Test LDA analysis with text preprocessing"""
   # Add some noisy data
   noisy_df = sample_tweets_df.copy()
   noisy_df['Concatenated Tweets'] = noisy_df['Concatenated Tweets'].apply(
       lambda x: x + ' !@#$% 123'
   )
   
   result = perform_lda_analysis(noisy_df)
   
   # Check that numbers and special characters are handled
   for keywords in result['LDA Keywords']:
       assert all(not word.isdigit() for word in keywords)
       assert all(not any(char in '!@#$%' for char in word) for word in keywords)