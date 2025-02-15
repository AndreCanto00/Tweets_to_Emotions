import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Tuple
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

class BertSemanticAnalyzer:
    """Class for performing semantic analysis using BERT embeddings."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize the BERT analyzer.
        
        Args:
            model_name (str): Name of the pre-trained BERT model to use
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set model to evaluation mode
        
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for a given text.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: BERT embedding vector
        """
        tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs['last_hidden_state'].mean(dim=1).squeeze().numpy()
    
    def calculate_tweet_antonym_similarities(self, 
                                          tweets_df: pd.DataFrame,
                                          antonyms_dict: Dict[str, List[str]]
                                          ) -> Dict[str, Dict[int, float]]:
        """
        Calculate similarities between tweets and their category antonyms.
        
        Args:
            tweets_df (pd.DataFrame): DataFrame containing tweets by category
            antonyms_dict (Dict[str, List[str]]): Dictionary of antonyms by category
            
        Returns:
            Dict[str, Dict[int, float]]: Nested dictionary of similarity scores
        """
        results = {}
        
        for category, antonyms in antonyms_dict.items():
            if not antonyms:
                continue
                
            # Get antonym embeddings
            category_embeddings = [self.get_bert_embedding(antonym) 
                                 for antonym in antonyms]
            
            # Get tweets for this category
            index = tweets_df.index[tweets_df['Sentiment Category'] == category].tolist()[0]
            records = tweets_df.at[index, 'Tweets']
            
            # Calculate similarities
            results[category] = {}
            for n, record in enumerate(records):
                record_embedding = self.get_bert_embedding(record)
                cosine_similarities = [
                    cosine_similarity([record_embedding], [antonym_embedding])[0][0]
                    for antonym_embedding in category_embeddings
                ]
                
                # Convert similarity to distance (1 - similarity)
                max_cosine_similarity = (
                    1 if not cosine_similarities 
                    else 1 - max(cosine_similarities)
                )
                results[category][n] = max_cosine_similarity
                
        return results
    
    def summarize_similarities(self, 
                             similarity_results: Dict[str, Dict[int, float]]
                             ) -> pd.DataFrame:
        """
        Create summary statistics for similarity results.
        
        Args:
            similarity_results (Dict[str, Dict[int, float]]): Results from 
                calculate_tweet_antonym_similarities
            
        Returns:
            pd.DataFrame: Summary statistics by category
        """
        results = []
        
        for sentiment_category, similarities in similarity_results.items():
            values = [sim for sim in similarities.values() 
                     if isinstance(sim, (float, np.ndarray))]
            
            if values:
                result = {
                    "Sentiment Category": sentiment_category,
                    "Min": min(values),
                    "Max": max(values),
                    "Mean": sum(values) / len(values),
                }
                results.append(result)
                
        return pd.DataFrame(results)

def analyze_semantic_similarities(tweets_df: pd.DataFrame,
                               antonyms_dict: Dict[str, List[str]],
                               model_name: str = 'bert-base-uncased'
                               ) -> Tuple[Dict[str, Dict[int, float]], pd.DataFrame]:
    """
    Perform complete semantic analysis using BERT.
    
    Args:
        tweets_df (pd.DataFrame): DataFrame containing tweets
        antonyms_dict (Dict[str, List[str]]): Dictionary of antonyms by category
        model_name (str): Name of BERT model to use
        
    Returns:
        Tuple[Dict, pd.DataFrame]: Raw similarities and summary statistics
    """
    analyzer = BertSemanticAnalyzer(model_name)
    similarities = analyzer.calculate_tweet_antonym_similarities(
        tweets_df, antonyms_dict
    )
    summary = analyzer.summarize_similarities(similarities)
    return similarities, summary